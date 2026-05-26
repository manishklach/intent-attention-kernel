from intent_attention.cost_model import savings_report
from intent_attention.synthetic_traces import generate_agentic_layout

def main():
    sizes = [1024, 4096, 16384, 65536]
    try:
        from tabulate import tabulate
        has_tabulate = True
    except ImportError:
        has_tabulate = False
        
    data = []
    headers = ["Total Tokens", "Selected Tokens", "Fraction", "FLOPs Saved %", "KV Bytes Saved %"]
    
    for size in sizes:
        layout = generate_agentic_layout(size)
        report = savings_report(1, 32, 128, size, layout.selected_token_count(), 128)
        data.append([
            size,
            layout.selected_token_count(),
            f"{report['selected_fraction']:.2f}",
            f"{report['flops_saved_pct']:.1f}%",
            f"{report['kv_bytes_saved_pct']:.1f}%"
        ])
        
    if has_tabulate:
        print(tabulate(data, headers=headers, tablefmt="github"))
    else:
        print(f"{headers[0]:<15} | {headers[1]:<18} | {headers[2]:<10} | {headers[3]:<15} | {headers[4]}")
        print("-" * 80)
        for row in data:
            print(f"{row[0]:<15} | {row[1]:<18} | {row[2]:<10} | {row[3]:<15} | {row[4]}")

if __name__ == "__main__":
    main()
