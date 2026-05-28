from intent_attention.cost_model import savings_report
from intent_attention.synthetic_traces import generate_agentic_layout


def main():
    sizes = [1024, 4096, 16384, 65536]

    try:
        from tabulate import tabulate

        headers = [
            "Total Tokens",
            "Selected",
            "Fraction",
            "FLOPs Saved %",
            "KV Bytes Saved %",
        ]
        rows = []
        for size in sizes:
            layout = generate_agentic_layout(size, seed=0)
            report = savings_report(
                1, 32, 128, size, layout.selected_token_count(), 128
            )
            rows.append(
                [
                    size,
                    layout.selected_token_count(),
                    f"{report['selected_fraction']:.4f}",
                    f"{report['flops_saved_pct']:.2f}%",
                    f"{report['kv_bytes_saved_pct']:.2f}%",
                ]
            )
        print(tabulate(rows, headers=headers, tablefmt="github"))
    except ImportError:
        print(
            f"{'Total':>12} {'Selected':>10} {'Fraction':>9} {'FLOPs%':>8} {'KV%':>8}"
        )
        print("-" * 52)
        for size in sizes:
            layout = generate_agentic_layout(size, seed=0)
            report = savings_report(
                1, 32, 128, size, layout.selected_token_count(), 128
            )
            print(
                f"{size:>12} {layout.selected_token_count():>10} "
                f"{report['selected_fraction']:>9.4f} "
                f"{report['flops_saved_pct']:>7.2f}% "
                f"{report['kv_bytes_saved_pct']:>7.2f}%"
            )


if __name__ == "__main__":
    main()
