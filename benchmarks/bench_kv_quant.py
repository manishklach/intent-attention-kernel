from __future__ import annotations


def main() -> None:
    kv_len_sizes = [4096, 16384, 65536]
    skip_ratios = [0.3, 0.5, 0.7]
    d_head = 128
    page_size = 128

    B = 1
    H = 32
    Q = 128

    total_pages_fp16_bytes = 0
    total_pages_int8_bytes = 0

    header = (
        f"{'KV tokens':>10} {'Skip ratio':>11} {'Selected':>10}"
        f" {'fp16 bytes':>14} {'int8+scale':>14} {'Saved %':>8}"
    )
    print(header)
    print("-" * 68)

    for kv_len in kv_len_sizes:
        total_tokens = kv_len
        num_pages = (total_tokens + page_size - 1) // page_size
        fp16_page = 2 * page_size * d_head * 2
        int8_page = page_size * d_head * 1
        scale_page = d_head * 2

        for skip in skip_ratios:
            selected_pages = max(1, int(num_pages * (1.0 - skip)))
            selected_tokens = selected_pages * page_size

            fp16_dense = 2 * total_tokens * d_head * 2 * H * B  # H*B for full KV
            int8_selected = (
                2 * selected_tokens * d_head * 1  # K+V int8
                + 2 * selected_pages * d_head * 2  # K+V scales fp16
            ) * H * B

            fp16_dense_b = int(round(fp16_dense / 1e6, 0))
            int8_selected_b = int(round(int8_selected / 1e6, 0))
            saved_pct = (1.0 - int8_selected / max(fp16_dense, 1)) * 100.0

            print(
                f"{kv_len:>10} {skip:>10.1f}  {selected_tokens:>8}"
                f" {fp16_dense_b:>12d} MB {int8_selected_b:>12d} MB {saved_pct:>7.1f}%"
            )


if __name__ == "__main__":
    main()
