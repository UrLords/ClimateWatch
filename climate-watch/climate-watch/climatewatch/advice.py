"""Community advice from a VERIFIED playbook. No free-form generation => no hallucinated civic advice.
Optional later: let an LLM only *rephrase* these actions in the local language."""

PLAYBOOK = {
    "drain_blocked": [
        "Clear debris from the drainage grate (wear gloves; avoid traffic).",
        "Report persistent blockage to the local public-works office.",
    ],
    "waste_pile": [
        "Move waste to the nearest official collection point.",
        "Report recurring dumping spots to the sanitation authority.",
    ],
    "flood_water": [
        "Avoid driving through standing water.",
        "Move valuables up and follow local flood alerts.",
    ],
    "open_burning": [
        "Do not burn waste in the open; call the local fire/environment hotline."
    ],
    "smoke": [
        "Reduce outdoor activity; consider a mask rated for fine particles (N95/KN95)."
    ],
    "haze_high": [
        "Air visibility is poor: limit outdoor exercise, keep windows closed, protect children and elders."
    ],
    "haze_moderate": [
        "Visibility is reduced: sensitive groups should limit prolonged outdoor activity."
    ],
}


def build_advice(
    haze_index: float,
    haze_conf: float,
    hazards: set[str],
    flood: dict,
    relative: bool = False,
) -> dict:
    hi, mod = (65, 40) if relative else (75, 55)
    keys = []
    if haze_conf >= 0.5:
        keys += (
            ["haze_high"]
            if haze_index >= hi
            else ["haze_moderate"]
            if haze_index >= mod
            else []
        )
    keys += [
        k
        for k in ("open_burning", "smoke", "flood_water", "drain_blocked", "waste_pile")
        if k in hazards
    ]
    actions, seen = [], set()
    for k in keys:
        for a in PLAYBOOK[k]:
            if a not in seen:
                seen.add(a)
                actions.append(a)
    if flood["level"] in ("medium", "high") and flood.get("rain"):
        actions.insert(
            0,
            f"Flood risk {flood['level']}: ~{flood['rain']['mm']} mm rain expected in the next hours.",
        )
    return {"source": "playbook", "actions": actions}
