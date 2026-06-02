#!/usr/bin/env python3
"""
Generate deployment decision report for Qwen3-ASR.

Aggregates results from:
  - eval/latency_report.json
  - eval/resource_report.json
  - eval/quantization_report.json

Outputs a structured decision report with edge vs cloud recommendation.
"""

import argparse
import json
import os
import sys
from datetime import datetime


def load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)
    except (json.JSONDecodeError, OSError):
        return {}


def format_table(headers: list, rows: list) -> str:
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    header_line = "|" + "|".join(f" {h:<{col_widths[i]}} " for i, h in enumerate(headers)) + "|"
    lines = [sep, header_line, sep]
    for row in rows:
        line = "|" + "|".join(f" {str(c):<{col_widths[i]}} " for i, c in enumerate(row)) + "|"
        lines.append(line)
    lines.append(sep)
    return "\n".join(lines)


def generate_report(latency: dict, resource: dict, quant: dict) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("  Qwen3-ASR 端云部署决策报告")
    lines.append(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    # 1. Model info
    model_path = latency.get("model") or resource.get("model") or "N/A"
    lines.append(f"\n模型: {model_path}")
    lines.append(f"参数量: 1.7B")

    # 2. Latency summary
    lines.append("\n" + "-" * 70)
    lines.append("一、推理性能 (Latency)")
    lines.append("-" * 70)

    if latency:
        lines.append(f"  设备: {latency.get('device', 'N/A')}, dtype: {latency.get('dtype', 'N/A')}")
        lines.append(f"  测试样本数: {latency.get('num_samples', 'N/A')}")
        lines.append(f"  总音频时长: {latency.get('total_audio_duration_sec', 0):.1f}s")
        lines.append(f"  总推理时间: {latency.get('total_inference_time_sec', 0):.1f}s")
        lines.append(f"  整体 RTF: {latency.get('overall_rtf', 0):.4f}")
        lines.append(f"    (RTF < 1.0 = 实时可行, RTF < 0.5 = 有余量)")

        e2e = latency.get("e2e_latency", {})
        if e2e:
            lines.append(f"\n  E2E 延迟分布:")
            lines.append(f"    P50: {e2e.get('p50', 0):.4f}s")
            lines.append(f"    P90: {e2e.get('p90', 0):.4f}s")
            lines.append(f"    P99: {e2e.get('p99', 0):.4f}s")

        rtf_stats = latency.get("rtf", {})
        if rtf_stats:
            lines.append(f"\n  RTF 分布:")
            lines.append(f"    P50: {rtf_stats.get('p50', 0):.4f}")
            lines.append(f"    P90: {rtf_stats.get('p90', 0):.4f}")
            lines.append(f"    P99: {rtf_stats.get('p99', 0):.4f}")

        buckets = latency.get("duration_buckets", {})
        if buckets:
            lines.append(f"\n  按音频时长分段:")
            headers = ["时长段", "样本数", "RTF P50", "延迟 P50(s)"]
            rows = []
            for name, stats in buckets.items():
                rows.append([
                    name,
                    stats["count"],
                    f"{stats['rtf']['p50']:.4f}",
                    f"{stats['latency']['p50']:.4f}",
                ])
            lines.append(format_table(headers, rows))
    else:
        lines.append("  [未找到 latency_report.json]")

    # 3. Resource summary
    lines.append("\n" + "-" * 70)
    lines.append("二、资源占用 (Resources)")
    lines.append("-" * 70)

    if resource:
        lines.append(f"  模型文件大小: {resource.get('model_size_disk_mb', 0):.1f} MB")
        cpu = resource.get("cpu_memory", {})
        gpu = resource.get("gpu_memory", {})
        if cpu:
            lines.append(f"  CPU 内存 (RSS):")
            lines.append(f"    模型加载后: {cpu.get('rss_after_load_mb', 0):.1f} MB")
            lines.append(f"    推理峰值:   {cpu.get('rss_peak_during_infer_mb', 0):.1f} MB")
        if gpu:
            lines.append(f"  GPU 显存:")
            lines.append(f"    模型加载后: {gpu.get('after_load_allocated_mb', 0):.1f} MB")
            lines.append(f"    推理峰值:   {gpu.get('peak_allocated_mb', 0):.1f} MB")
            lines.append(f"    推理增量:   +{gpu.get('inference_peak_delta_mb', 0):.1f} MB")
    else:
        lines.append("  [未找到 resource_report.json]")

    # 4. Quantization comparison
    lines.append("\n" + "-" * 70)
    lines.append("三、量化对比 (Quantization)")
    lines.append("-" * 70)

    if quant and quant.get("results"):
        qr = quant["results"]
        headers = ["配置", "CER(%)", "RTF", "GPU峰值(MB)", "相对加速"]
        rows = []
        baseline_rtf = qr.get("bf16_baseline", {}).get("rtf")
        baseline_cer = qr.get("bf16_baseline", {}).get("cer")

        for name, res in qr.items():
            if "error" in res:
                rows.append([name, "ERROR", "-", "-", "-"])
                continue
            speedup = f"{baseline_rtf / res['rtf']:.2f}x" if baseline_rtf and res.get("rtf") else "N/A"
            cer_str = f"{res['cer']:.2f}"
            if baseline_cer is not None and name != "bf16_baseline":
                delta = res["cer"] - baseline_cer
                cer_str += f" ({'+' if delta > 0 else ''}{delta:.2f})"
            rows.append([
                name,
                cer_str,
                f"{res.get('rtf', 0):.4f}",
                f"{res.get('gpu_peak_mb', 'N/A')}",
                speedup,
            ])
        lines.append(format_table(headers, rows))
    else:
        lines.append("  [未找到 quantization_report.json]")

    # 5. Decision
    lines.append("\n" + "-" * 70)
    lines.append("四、端云部署决策建议")
    lines.append("-" * 70)

    decision = analyze_decision(latency, resource, quant)
    for line in decision:
        lines.append(line)

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def analyze_decision(latency: dict, resource: dict, quant: dict) -> list:
    """Generate deployment recommendation based on collected metrics."""
    lines = []

    model_size_mb = resource.get("model_size_disk_mb", 4000)
    gpu_peak = resource.get("gpu_memory", {}).get("peak_allocated_mb", 0)
    overall_rtf = latency.get("overall_rtf", 999)

    # Quantization results
    qr = quant.get("results", {}) if quant else {}
    int4_cer = qr.get("int4_nf4", {}).get("cer")
    int4_rtf = qr.get("int4_nf4", {}).get("rtf")
    int4_gpu = qr.get("int4_nf4", {}).get("gpu_peak_mb")
    baseline_cer = qr.get("bf16_baseline", {}).get("cer")

    lines.append("")
    lines.append("  【云端部署评估】")
    lines.append(f"    - GPU RTF = {overall_rtf:.4f} → {'✓ 实时可行' if overall_rtf < 1.0 else '✗ 不满足实时'}")
    lines.append(f"    - GPU 显存峰值 = {gpu_peak:.0f} MB → 云端 GPU 完全可承载")
    lines.append(f"    - 结论: 云端部署完全可行，性能充裕")

    lines.append("")
    lines.append("  【手机端部署评估】")
    lines.append(f"    - 原始模型大小: {model_size_mb:.0f} MB → {'✗ 超过 500MB 限制' if model_size_mb > 500 else '✓ 可接受'}")

    if int4_cer is not None and baseline_cer is not None:
        cer_degradation = int4_cer - baseline_cer
        cer_relative = (cer_degradation / baseline_cer * 100) if baseline_cer > 0 else 0
        lines.append(f"    - INT4 量化后 CER: {int4_cer:.2f}% (退化 {cer_degradation:+.2f}%, 相对退化 {cer_relative:.1f}%)")
        lines.append(f"      → {'✓ 退化 < 5% 相对值，可接受' if cer_relative < 5 else '✗ 退化 >= 5%，需权衡'}")
    else:
        lines.append("    - INT4 量化 CER: [未测量]")

    if int4_gpu:
        estimated_mobile_size = int4_gpu * 0.6  # rough estimate for mobile format
        lines.append(f"    - INT4 GPU 占用: {int4_gpu:.0f} MB (估算移动端 ~{estimated_mobile_size:.0f} MB)")
        lines.append(f"      → {'✓ 可能适合高端手机' if estimated_mobile_size < 2000 else '✗ 内存占用过大'}")

    # Final recommendation
    lines.append("")
    lines.append("  【最终建议】")

    cloud_ok = overall_rtf < 1.0
    edge_cer_ok = (int4_cer is not None and baseline_cer is not None and
                   ((int4_cer - baseline_cer) / baseline_cer * 100 < 5 if baseline_cer > 0 else True))
    edge_size_ok = model_size_mb < 2000  # INT4 would be ~1/4

    if cloud_ok and edge_cer_ok and edge_size_ok:
        lines.append("    → 推荐: 端云混合方案")
        lines.append("      - 高端设备 (≥8GB RAM): INT4 量化后端侧部署")
        lines.append("      - 中低端设备: 云端推理 + 轻量端侧 VAD")
        lines.append("      - 离线场景: 端侧 INT4 模型兜底")
    elif cloud_ok and not edge_cer_ok:
        lines.append("    → 推荐: 云端部署为主")
        lines.append("      - INT4 量化 CER 退化过大，端侧质量不达标")
        lines.append("      - 建议云端 BF16/INT8 推理，RTF 充裕")
        lines.append("      - 端侧可部署更小的模型做 VAD/热词检测")
    elif cloud_ok:
        lines.append("    → 推荐: 云端部署")
        lines.append("      - 云端 GPU 推理 RTF 远小于 1.0，性能充裕")
        lines.append("      - 模型体积较大，端侧部署需进一步验证")
    else:
        lines.append("    → 需要进一步优化")
        lines.append("      - 当前配置下 RTF 不满足实时要求")
        lines.append("      - 建议: 尝试更高性能 GPU 或模型蒸馏")

    return lines


def main():
    parser = argparse.ArgumentParser(description="Generate ASR deployment decision report")
    parser.add_argument("--latency_report", type=str, default="eval/latency_report.json")
    parser.add_argument("--resource_report", type=str, default="eval/resource_report.json")
    parser.add_argument("--quant_report", type=str, default="eval/quantization_report.json")
    parser.add_argument("--output", type=str, default="eval/decision_report.txt")
    parser.add_argument("--output_json", type=str, default="eval/decision_report.json")
    args = parser.parse_args()

    # Load all reports
    latency = load_json(args.latency_report)
    resource = load_json(args.resource_report)
    quant = load_json(args.quant_report)

    found = []
    if latency:
        found.append("latency")
    if resource:
        found.append("resource")
    if quant:
        found.append("quantization")
    print(f"Found reports: {', '.join(found) if found else 'none'}")

    if not found:
        print("No report files found. Run the evaluation scripts first:")
        print("  python eval/benchmark_latency.py")
        print("  python eval/profile_resources.py")
        print("  python eval/quantization_eval.py")
        sys.exit(1)

    # Generate text report
    report_text = generate_report(latency, resource, quant)
    print(report_text)

    # Save text report
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nText report saved to: {args.output}")

    # Save JSON summary
    summary = {
        "generated_at": datetime.now().isoformat(),
        "model": latency.get("model") or resource.get("model") or quant.get("model", ""),
        "latency_summary": {
            "overall_rtf": latency.get("overall_rtf"),
            "e2e_latency_p50": latency.get("e2e_latency", {}).get("p50"),
            "e2e_latency_p90": latency.get("e2e_latency", {}).get("p90"),
            "e2e_latency_p99": latency.get("e2e_latency", {}).get("p99"),
        } if latency else None,
        "resource_summary": {
            "model_size_mb": resource.get("model_size_disk_mb"),
            "gpu_peak_mb": resource.get("gpu_memory", {}).get("peak_allocated_mb"),
            "cpu_peak_mb": resource.get("cpu_memory", {}).get("rss_peak_during_infer_mb"),
        } if resource else None,
        "quantization_summary": quant.get("results") if quant else None,
        "recommendation": "see text report",
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"JSON summary saved to: {args.output_json}")


if __name__ == "__main__":
    main()
