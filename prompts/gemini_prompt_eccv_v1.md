# Verbatim Gemini API prompt (eccv_v1)

The exact prompt sent to the Gemini APIs for the paper's baselines
(supplementary Sec. 5). Decoding uses temperature 0; on a JSON-parse
failure the call is retried up to 3 times with temperature stepped by
+0.3 per attempt. Only the JSON is scored; any prose is discarded.

```text
You are analyzing ONE sports photograph. Return ONLY a valid JSON object,
no prose, no markdown fences.

CRITICAL ANTI-HALLUCINATION RULES:
- Do NOT use outside knowledge. Do NOT guess player names, team names,
  leagues, game context, or famous kits.
- Only output what is directly visible in the image. If a value is not
  clearly readable/visible, set it to null (or [] for lists). Never invent.

OUTPUT JSON SCHEMA:
{
  "sport_type": "Basketball | Soccer | Hockey | American Football |
                 Baseball | Tennis | Rugby | Cricket | Volleyball | Other",
  "scene_type": "In-Game | Warm-ups | Player Arrivals | Press Conference |
                 Interview | Locker Room | Winning Ceremony | Other",
  "description": "<one concise sentence; no real names/teams>",
  "players": [
    {
      "player_index": <1-based index>,
      "team_affiliation": "Team A | Team B | Other",
      "bbox": {"x": <left 0-1>, "y": <top 0-1>, "w": <0-1>, "h": <0-1>},
      "jersey_facing": "Front | Back | Side",
      "jersey_number": "<clearly legible digits only, else null>",
      "ocr_items": [ {"text": "<readable text ON this player>"} ]
    }
  ]
}

RULES:
- DETECTION SCOPE: detect ATHLETES only; exclude referees, officials,
  coaches, staff, and off-field people (crowd/fans, bench staff).
- bbox: tight box around the whole visible body, normalized 0-1, origin
  top-left; x,y are the top-left corner, w,h are width/height.
- team_affiliation: cluster the competing athletes into "Team A"/"Team B"
  strictly by kit/jersey-color similarity; use "Other" ONLY when the team
  is genuinely unclear. Be consistent. Do NOT map real-world team names.
- jersey_number: ONLY if clearly legible; digits only; else null.
- ocr_items: EVERY clearly readable text fragment physically on the player
  (include partial). Never invent text; empty list if nothing readable.
- Output strictly the JSON object. No commentary, no markdown.
```
