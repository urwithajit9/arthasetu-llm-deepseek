#!/usr/bin/env python3
"""
ArthaSeetu Brain - Cost Calculator
Calculate and optimize Modal GPU costs
"""
import argparse

# ============================================================================
# GPU PRICING (as of 2024)
# ============================================================================
GPU_PRICES = {
    "A10G": 1.10,
    "L4": 0.60,
    "T4": 0.40,
}

# ============================================================================
# COST CALCULATOR
# ============================================================================
def calculate_monthly_cost(
    gpu_type: str,
    keep_warm: int,
    avg_scale_up_hours: float,
    avg_scaled_containers: float,
):
    """Calculate estimated monthly cost"""

    hourly_rate = GPU_PRICES[gpu_type]

    # Keep-warm cost (24/7)
    keep_warm_cost = keep_warm * hourly_rate * 24 * 30

    # Scale-up cost (additional containers during peak)
    scale_up_cost = avg_scale_up_hours * avg_scaled_containers * hourly_rate * 30

    total_monthly = keep_warm_cost + scale_up_cost

    return {
        "gpu_type": gpu_type,
        "hourly_rate": hourly_rate,
        "keep_warm_containers": keep_warm,
        "keep_warm_cost_monthly": keep_warm_cost,
        "scale_up_hours_per_day": avg_scale_up_hours,
        "avg_scaled_containers": avg_scaled_containers,
        "scale_up_cost_monthly": scale_up_cost,
        "total_monthly_cost": total_monthly,
        "total_yearly_cost": total_monthly * 12,
    }

def compare_gpus(keep_warm: int, usage_hours: float, scaled_containers: float):
    """Compare costs across different GPUs"""

    print("=" * 80)
    print("💰 GPU Cost Comparison")
    print("=" * 80)
    print()
    print(f"Configuration:")
    print(f"  • Keep-warm containers: {keep_warm}")
    print(f"  • Peak usage hours/day: {usage_hours}")
    print(f"  • Additional containers during peak: {scaled_containers}")
    print()

    results = []

    for gpu in ["A10G", "L4", "T4"]:
        cost = calculate_monthly_cost(gpu, keep_warm, usage_hours, scaled_containers)
        results.append(cost)

    print("-" * 80)
    print(f"{'GPU':<10} {'Hourly':<12} {'Monthly':<15} {'Yearly':<15} {'Savings'}")
    print("-" * 80)

    baseline = results[0]["total_monthly_cost"]

    for cost in results:
        savings = baseline - cost["total_monthly_cost"]
        savings_pct = (savings / baseline) * 100 if baseline > 0 else 0

        print(
            f"{cost['gpu_type']:<10} "
            f"${cost['hourly_rate']:<11.2f} "
            f"${cost['total_monthly_cost']:<14.2f} "
            f"${cost['total_yearly_cost']:<14.2f} "
            f"-${savings:.2f} ({savings_pct:+.0f}%)"
        )

    print("-" * 80)
    print()

    # Best choice
    best = min(results, key=lambda x: x["total_monthly_cost"])
    print(f"🏆 Best Choice: {best['gpu_type']} (${best['total_monthly_cost']:.2f}/month)")
    print()

def optimization_scenarios():
    """Show different optimization scenarios"""

    print("=" * 80)
    print("🔧 Cost Optimization Scenarios")
    print("=" * 80)
    print()

    scenarios = [
        {
            "name": "Original (A10G, Always Warm)",
            "gpu": "A10G",
            "keep_warm": 1,
            "usage_hours": 8,
            "scaled": 1,
        },
        {
            "name": "Optimized (L4, Always Warm)",
            "gpu": "L4",
            "keep_warm": 1,
            "usage_hours": 8,
            "scaled": 1,
        },
        {
            "name": "Cost-Conscious (L4, Peak Warm)",
            "gpu": "L4",
            "keep_warm": 0,
            "usage_hours": 8,
            "scaled": 2,
        },
        {
            "name": "Budget (T4, On-Demand)",
            "gpu": "T4",
            "keep_warm": 0,
            "usage_hours": 6,
            "scaled": 1,
        },
    ]

    results = []

    for scenario in scenarios:
        cost = calculate_monthly_cost(
            scenario["gpu"],
            scenario["keep_warm"],
            scenario["usage_hours"],
            scenario["scaled"],
        )
        cost["name"] = scenario["name"]
        results.append(cost)

    print("-" * 80)
    print(f"{'Scenario':<30} {'GPU':<8} {'Monthly':<15} {'Cold Starts'}")
    print("-" * 80)

    for cost in results:
        cold_starts = "None" if cost["keep_warm_containers"] > 0 else "~30s"
        print(
            f"{cost['name']:<30} "
            f"{cost['gpu_type']:<8} "
            f"${cost['total_monthly_cost']:<14.2f} "
            f"{cold_starts}"
        )

    print("-" * 80)
    print()

    # Savings analysis
    baseline = results[0]["total_monthly_cost"]
    best = min(results, key=lambda x: x["total_monthly_cost"])
    savings = baseline - best["total_monthly_cost"]

    print(f"💡 Potential Monthly Savings: ${savings:.2f} ({savings/baseline*100:.0f}%)")
    print(f"💡 Potential Yearly Savings: ${savings * 12:.2f}")
    print()

def usage_pattern_calculator():
    """Interactive cost calculator based on usage pattern"""

    print("=" * 80)
    print("📊 Usage Pattern Calculator")
    print("=" * 80)
    print()

    patterns = {
        "1": {
            "name": "Light Usage (Personal/Dev)",
            "requests_per_day": 100,
            "peak_hours": 4,
            "avg_response_time": 2.0,
        },
        "2": {
            "name": "Medium Usage (Small Team)",
            "requests_per_day": 1000,
            "peak_hours": 8,
            "avg_response_time": 1.5,
        },
        "3": {
            "name": "Heavy Usage (Production)",
            "requests_per_day": 10000,
            "peak_hours": 12,
            "avg_response_time": 1.0,
        },
    }

    print("Select usage pattern:")
    for key, pattern in patterns.items():
        print(f"  {key}. {pattern['name']} ({pattern['requests_per_day']:,} req/day)")
    print()

    choice = input("Enter choice (1-3): ").strip()

    if choice not in patterns:
        print("Invalid choice")
        return

    pattern = patterns[choice]

    print()
    print(f"📈 Analysis for: {pattern['name']}")
    print(f"  • Requests/day: {pattern['requests_per_day']:,}")
    print(f"  • Peak hours: {pattern['peak_hours']}")
    print(f"  • Avg response time: {pattern['avg_response_time']}s")
    print()

    # Calculate required containers
    requests_per_hour_peak = pattern['requests_per_day'] / pattern['peak_hours']
    requests_per_second = requests_per_hour_peak / 3600
    containers_needed = max(1, int(requests_per_second * pattern['avg_response_time'] / 10))

    print(f"📊 Recommendations:")
    print(f"  • Peak requests/second: {requests_per_second:.2f}")
    print(f"  • Recommended containers: {containers_needed}")
    print()

    # Cost estimates for different GPUs
    print("💰 Estimated Monthly Costs:")
    print()

    for gpu in ["L4", "T4"]:
        cost = calculate_monthly_cost(
            gpu,
            keep_warm=1 if containers_needed > 0 else 0,
            avg_scale_up_hours=pattern['peak_hours'],
            avg_scaled_containers=max(0, containers_needed - 1),
        )

        print(f"  {gpu}:")
        print(f"    • Monthly: ${cost['total_monthly_cost']:.2f}")
        print(f"    • Per request: ${cost['total_monthly_cost'] / (pattern['requests_per_day'] * 30):.4f}")
        print()

def quick_estimate():
    """Quick cost estimate"""

    print("=" * 80)
    print("⚡ Quick Cost Estimate")
    print("=" * 80)
    print()

    print("For typical production setup (L4, 1 warm container):")
    print()

    cost = calculate_monthly_cost("L4", keep_warm=1, avg_scale_up_hours=8, avg_scaled_containers=1)

    print(f"  • Base (1 warm container 24/7):    ${cost['keep_warm_cost_monthly']:.2f}/month")
    print(f"  • Peak usage (8hrs/day, +1 container): ${cost['scale_up_cost_monthly']:.2f}/month")
    print(f"  • Total estimated:                  ${cost['total_monthly_cost']:.2f}/month")
    print()
    print(f"  💡 vs A10G: Save ${(1.10 - 0.60) * 24 * 30:.2f}/month (45%)")
    print()

# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="ArthaSeetu Brain Cost Calculator")
    parser.add_argument(
        "--mode",
        choices=["compare", "optimize", "usage", "quick"],
        default="quick",
        help="Calculator mode"
    )
    parser.add_argument("--keep-warm", type=int, default=1, help="Keep-warm containers")
    parser.add_argument("--peak-hours", type=float, default=8, help="Peak hours per day")
    parser.add_argument("--scaled-containers", type=float, default=1, help="Additional containers during peak")

    args = parser.parse_args()

    if args.mode == "compare":
        compare_gpus(args.keep_warm, args.peak_hours, args.scaled_containers)
    elif args.mode == "optimize":
        optimization_scenarios()
    elif args.mode == "usage":
        usage_pattern_calculator()
    else:
        quick_estimate()

if __name__ == "__main__":
    main()