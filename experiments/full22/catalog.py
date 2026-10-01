"""Finite, recorded experiment configurations, before inspecting outcomes."""

def configurations():
    result = []
    def add(name, ids, **config):
        result.append({"name": name, "principles": ids, "config": config})
    for bits in [2, 3, 4, 6, 8]:
        add(f"q{bits}-maxabs", [5], bits=bits, clip=False)
        add(f"q{bits}-fit", [5], bits=bits)
    add("ternary-maxabs", [5], bits=2, alphabet="ternary", clip=False)
    add("ternary-fit", [5], bits=2, alphabet="ternary")
    add("nonuniform-four", [5], bits=2, alphabet="nonuniform4")
    for bits in [3, 4]:
        for rounding, principle in [("adjacent", 3), ("feedback", 3), ("joint", 4)]:
            add(f"q{bits}-{rounding}", [principle], bits=bits, rounding=rounding)
        for group in [16, 64, 128]:
            add(f"q{bits}-g{group}", [5, 16, 17], bits=bits, group=group)
        add(f"q{bits}-scale-f32", [16], bits=bits, scale_dtype="f32")
        for transform, settings in [
            ("h8", {"rotation": 8}), ("signed-h32", {"rotation": 32, "signed": True}),
            ("diagonal", {"diagonal": 0.5}),
            ("diagonal-h32", {"diagonal": 0.5, "rotation": 32, "signed": True}),
        ]:
            add(f"q{bits}-{transform}", [6], bits=bits, **settings)
        add(f"q{bits}-precondition-feedback", [3, 6], bits=bits, rounding="feedback", diagonal=0.5, rotation=32, signed=True)
        for post in ["norm", "gain", "gain_dc"]:
            add(f"q{bits}-{post}", [12], bits=bits, post=post)
        add(f"q{bits}-feedback-gain", [3, 12], bits=bits, rounding="feedback", post="gain")
        add(f"q{bits}-posterior", [14], bits=bits, posterior=True)
        for rank in [4, 8]:
            add(f"q{bits}-rank{rank}", [10], bits=bits, rank=rank)
        for fraction in [0.001, 0.01]:
            for rule in ["magnitude", "sensitivity"]:
                add(f"q{bits}-escape-{rule}-{fraction}", [11], bits=bits, exceptions=fraction, exception_rule=rule)
        for price in [0.001, 0.005, 0.02]:
            add(f"q{bits}-adaptive-{price}", [17], family="adaptive", bits=bits, partition_lambda=price)
    add("e8", [7], family="e8")
    add("e8-signed-h32", [6, 7], family="e8", rotation=32, signed=True)
    add("e8-diagonal-h32", [6, 7], family="e8", diagonal=0.5, rotation=32, signed=True)
    for bits in [2, 3]:
        add(f"trellis-{bits}", [8], family="trellis", bits=bits)
        add(f"trellis-{bits}-h32", [6, 8], family="trellis", bits=bits, rotation=32, signed=True)
    for books in [2, 3, 4]:
        add(f"additive-{books}", [9], family="additive", books=books)
    add("additive-2-h32", [6, 9], family="additive", books=2, rotation=32, signed=True)
    add("additive-3-h32", [6, 9], family="additive", books=3, rotation=32, signed=True)
    for repair in ["flat", "row", "graph"]:
        add(f"q4-repair-{repair}", [13], bits=4, repair=repair, repair_strength=0.05)
    add("q4-graph-shuffled", [13], bits=4, repair="graph", repair_strength=0.05, shuffled_graph=True)
    for rounding in ["stochastic", "paired"]:
        for seed in [0, 1, 2]:
            add(f"q3-{rounding}-{seed}", [15], bits=3, rounding=rounding, clip=False, seed=seed)
    add("lossless", [16], family="lossless")
    return result


TENSORS = ["token_embd.weight", "blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_output.weight",
    "blk.3.ffn_up.weight", "blk.9.ffn_gate.weight", "blk.17.ffn_down.weight", "blk.29.ffn_down.weight"]
