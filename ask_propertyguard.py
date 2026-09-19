"""
Ask PropertyGuard — Conversational AI Engine
---------------------------------------------
Assembles a structured property context from PropertyGuard's existing ML and
climate outputs, then answers home-buyer questions in plain language.

Architecture
------------
  build_property_context(...)   Build a plain-text context string + a structured
                                data dict from existing app outputs.
  call_llm(...)                 Modular LLM hook — returns None by default.
  _rule_based_response(...)     Keyword-matched fallback grounded in structured data.
  get_chat_response(...)        Public entry point called from app.py.

No external API credentials, SDK imports, or network calls are required.
To connect an LLM, implement call_llm() — see the docstring inside that function.
"""

from typing import Optional


# ── System Prompt (used when an LLM is connected) ────────────────────────────
SYSTEM_PROMPT = """You are PropertyGuard Assistant, an AI advisor helping home buyers \
in India understand property valuations and climate sustainability analysis.

Your role:
- Explain results clearly in simple language suitable for a first-time home buyer.
- Use ONLY the property data provided in the context. Do not invent numbers.
- Distinguish between ML model predictions (estimates) and factual property data.
- When information is unavailable, say so honestly.
- Keep answers concise (3-5 sentences) unless more detail is genuinely needed.
- Never speculate about future property prices beyond what the data supports.

Always be helpful, honest, and grounded in the provided property data."""


# ── Suggested starter questions shown in the UI ──────────────────────────────
SUGGESTED_QUESTIONS = [
    "Why did this property receive this sustainability score?",
    "What climate risks should I check before buying?",
    "What questions should I ask the seller?",
    "Is this property suitable from a sustainability perspective?",
    "How can I improve this property's sustainability?",
    "Explain the property analysis in simple language.",
]


def build_property_context(
    city: str,
    locality: str,
    bedrooms: int,
    area_sqft: int,
    is_resale: str,
    amenities_dict: dict,
    pred_price: float,
    loc_data: Optional[dict],
    climate_info: dict,
) -> tuple:
    """
    Returns (context_string, structured_data).

    context_string  — plain text injected into the LLM prompt.
    structured_data — Python dict used directly by _rule_based_response so it
                      never needs to parse the text string.

    Nothing is recalculated here — every parameter is taken directly from
    variables already computed in app.py.
    """
    price_per_sqft = pred_price / area_sqft
    min_est = pred_price * 0.90
    max_est = pred_price * 1.10

    # ── Price formatting ──────────────────────────────────────────────────────
    if pred_price >= 10_000_000:
        price_fmt = f"₹{pred_price / 10_000_000:.2f} Crore"
    else:
        price_fmt = f"₹{pred_price / 100_000:.2f} Lakhs"

    # ── Locality benchmark ────────────────────────────────────────────────────
    diff_pct = None  # float | None; set inside if-branch below
    if loc_data:
        diff_pct = (
            (price_per_sqft - loc_data["median_price_sqft"])
            / loc_data["median_price_sqft"]
            * 100
        )
        direction = "above" if diff_pct > 0 else "below"
        benchmark_text = (
            f"Locality median price: ₹{loc_data['median_price'] / 100_000:.2f} Lakhs | "
            f"Locality median price/sqft: ₹{loc_data['median_price_sqft']:,.0f} | "
            f"Locality avg area: {loc_data['mean_area']:.0f} sqft | "
            f"This property is {direction} the locality median by {abs(diff_pct):.1f}%."
        )
    else:
        benchmark_text = "Locality benchmark data not available in the current dataset."

    # ── Active amenities ──────────────────────────────────────────────────────
    active_amenities = [k for k, v in amenities_dict.items() if v == 1]
    amenities_text = ", ".join(active_amenities) if active_amenities else "None selected"

    # ── Climate metrics (read directly from existing climate_info) ────────────
    m = climate_info["metrics"]
    score = climate_info["sustainability_score"]
    score_label = "High" if score >= 75 else ("Moderate" if score >= 55 else "Low")

    # ── Advice summary for context string ────────────────────────────────────
    advice_lines = "\n".join(
        f"  [{a['status'].upper()}] {a['icon']} {a['title']} ({a['category']}): {a['description']}"
        for a in climate_info["advice"]
    )

    context_string = f"""=== PROPERTY CONTEXT (PropertyGuard Analysis) ===

PROPERTY DETAILS:
- City: {city}
- Locality: {locality}
- Configuration: {bedrooms} BHK | {area_sqft:,} sqft | {is_resale}
- Active Amenities ({len(active_amenities)} selected): {amenities_text}

ML PRICE PREDICTION (ensemble model estimate — not a certified valuation):
- Predicted Market Value: {price_fmt}
- Price per Sq.Ft.: ₹{price_per_sqft:,.0f}
- Fair Value Range (±10%): ₹{min_est / 100_000:.1f}L – ₹{max_est / 100_000:.1f}L
- {benchmark_text}

SUSTAINABILITY & LIVING SCORE: {score}/100 ({score_label})
Score breakdown (each pillar weighted equally at 25%):
  ☀️  Solar Energy Potential:    {m['solar_score']}/100  (Solar insolation: {m['solar_insolation']} kWh/m²/day)
  🌡️  Thermal & Heat Comfort:    {m['thermal_score']}/100  (Peak summer temp: {m['summer_max_temp_c']}°C)
  🌿  Air Quality Index:          {m['aqi_score']}/100   (AQI: {m['aqi']} — {m['air_quality_category']})
  💧  Water & Flood Resilience:  {m['water_score']}/100  (Flood risk: {m['flood_risk']} | Water stress: {m['water_stress']})

NASA POWER CLIMATE INDICATORS:
- Annual Precipitation: {m['annual_precipitation_mm']} mm/year
- Average Humidity: {m['humidity_pct']}%
- Green Cover: {m['green_cover_pct']}%

EXISTING BUYER ADVICE (generated by PropertyGuard):
{advice_lines}

=== END OF PROPERTY CONTEXT ==="""

    structured_data = {
        # property
        "city": city,
        "locality": locality,
        "bedrooms": bedrooms,
        "area_sqft": area_sqft,
        "is_resale": is_resale,
        "active_amenities": active_amenities,
        # price
        "price_fmt": price_fmt,
        "price_per_sqft": price_per_sqft,
        "min_est_l": min_est / 100_000,
        "max_est_l": max_est / 100_000,
        "benchmark_text": benchmark_text,
        "diff_pct": diff_pct,
        "loc_data": loc_data,
        # sustainability score
        "score": score,
        "score_label": score_label,
        # climate pillars — values taken directly from climate_info["metrics"]
        "solar_score": m["solar_score"],
        "solar_insolation": m["solar_insolation"],
        "thermal_score": m["thermal_score"],
        "summer_max_temp_c": m["summer_max_temp_c"],
        "aqi_score": m["aqi_score"],
        "aqi": m["aqi"],
        "air_quality_category": m["air_quality_category"],
        "water_score": m["water_score"],
        "flood_risk": m["flood_risk"],
        "water_stress": m["water_stress"],
        # other climate
        "annual_precipitation_mm": m["annual_precipitation_mm"],
        "humidity_pct": m["humidity_pct"],
        "green_cover_pct": m["green_cover_pct"],
        # existing advice cards, unmodified
        "advice": climate_info["advice"],
    }

    return context_string, structured_data


def call_llm(
    system_prompt: str,
    property_context: str,
    user_question: str,
    chat_history: list,
) -> Optional[str]:
    """
    Modular LLM hook.

    Returns a string response, or None to fall back to the rule-based engine.

    To connect an LLM, replace `return None` with your API call.
    The function receives everything needed:
      system_prompt      — role and behaviour instructions
      property_context   — full plain-text property data block
      user_question      — current message from the user
      chat_history       — list of {"role": "user"|"assistant", "content": str}

    Example integrations (not active — implement your own):
    ─────────────────────────────────────────────────────────
    # IBM watsonx.ai (Granite):
    #   import os
    #   from ibm_watsonx_ai import APIClient, Credentials
    #   from ibm_watsonx_ai.foundation_models import ModelInference
    #   credentials = Credentials(url=os.getenv("WATSONX_URL"), api_key=os.getenv("WATSONX_API_KEY"))
    #   client = APIClient(credentials, project_id=os.getenv("WATSONX_PROJECT_ID"))
    #   model = ModelInference(model_id="ibm/granite-13b-instruct-v2", api_client=client)
    #   prompt = f"{system_prompt}\\n\\n{property_context}\\nUser: {user_question}\\nAssistant:"
    #   return model.generate_text(prompt=prompt).strip()

    # OpenAI / Azure OpenAI:
    #   import openai
    #   messages = [{"role": "system", "content": system_prompt + "\\n\\n" + property_context}]
    #   messages += [{"role": m["role"], "content": m["content"]} for m in chat_history[-6:]]
    #   messages.append({"role": "user", "content": user_question})
    #   resp = openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
    #   return resp.choices[0].message.content.strip()

    # Ollama (local):
    #   import requests
    #   payload = {"model": "llama3", "prompt": f"{system_prompt}\\n{property_context}\\n{user_question}"}
    #   r = requests.post("http://localhost:11434/api/generate", json=payload)
    #   return r.json()["response"].strip()
    """
    return None  # placeholder — rule-based fallback is used automatically


def _advice_for(advice_list: list, category_keyword: str) -> list:
    """Return existing advice cards whose category contains the keyword."""
    return [
        a for a in advice_list
        if category_keyword.lower() in a["category"].lower()
    ]


def _format_advice_cards(cards: list) -> str:
    """Format a subset of existing advice cards as readable bullet points."""
    if not cards:
        return ""
    return "\n".join(
        f"- **[{a['status'].upper()}]** {a['icon']} {a['title']}: {a['description']}"
        for a in cards
    )


def _rule_based_response(user_question: str, d: dict) -> str:
    """
    Keyword-matched fallback responses grounded entirely in the structured_data
    dict produced by build_property_context(). No string parsing, no invented
    numbers — every value comes directly from PropertyGuard's existing outputs.
    """
    q = user_question.lower().strip()

    # ── Intent: sustainability score / suitability ────────────────────────────
    if any(w in q for w in [
        "sustainability", "score", "pillar", "what does the score",
        "explain score", "living score", "how is the score",
        "suitable", "suitability"
    ]):
        solar_cards   = _format_advice_cards(_advice_for(d["advice"], "Solar"))
        water_cards   = _format_advice_cards(_advice_for(d["advice"], "Water"))
        thermal_cards = _format_advice_cards(_advice_for(d["advice"], "Thermal"))
        aqi_cards     = _format_advice_cards(_advice_for(d["advice"], "Air"))

        return (
            f"**Sustainability & Living Score — {d['score']}/100 ({d['score_label']})**\n\n"
            "PropertyGuard calculates this score from four equally weighted NASA POWER "
            "climate pillars (25% each):\n\n"
            f"- ☀️ **Solar Energy Potential: {d['solar_score']}/100** — "
            f"Solar insolation for {d['city']} is {d['solar_insolation']} kWh/m²/day.\n"
            + (f"  {solar_cards}\n" if solar_cards else "")
            + f"- 🌡️ **Thermal & Heat Comfort: {d['thermal_score']}/100** — "
            f"Peak summer temperature is {d['summer_max_temp_c']}°C.\n"
            + (f"  {thermal_cards}\n" if thermal_cards else "")
            + f"- 🌿 **Air Quality Index: {d['aqi_score']}/100** — "
            f"Average AQI is {d['aqi']} ({d['air_quality_category']}).\n"
            + (f"  {aqi_cards}\n" if aqi_cards else "")
            + f"- 💧 **Water & Flood Resilience: {d['water_score']}/100** — "
            f"Flood risk: {d['flood_risk']} | Water stress: {d['water_stress']}.\n"
            + (f"  {water_cards}\n" if water_cards else "")
            + "\nSee the **NASA Climate & Sustainability** tab for the full visual breakdown."
        )

    # ── Intent: price / valuation ─────────────────────────────────────────────
    elif any(w in q for w in [
        "price", "value", "cost", "worth", "valuation", "predict", "how much",
        "market value", "price per sqft"
    ]):
        benchmark_note = (
            f"\n📍 **Locality context:** {d['benchmark_text']}"
            if d["loc_data"] else
            "\n📍 Locality benchmark data is not available for this specific locality."
        )
        return (
            f"**ML Price Prediction — {d['price_fmt']}**\n\n"
            "This estimate comes from PropertyGuard's ensemble ML model "
            "(CatBoost + XGBoost + HistGradientBoosting), trained on real property "
            f"listings across 6 Indian cities. It factors in the property area "
            f"({d['area_sqft']:,} sqft), {d['bedrooms']} bedrooms, locality, city, "
            f"resale status, and your selected amenities.\n\n"
            f"**Fair value range (±10%):** ₹{d['min_est_l']:.1f}L – ₹{d['max_est_l']:.1f}L"
            f"{benchmark_note}\n\n"
            "⚠️ This is a **model estimate, not a certified valuation**. "
            "Verify with a RERA-registered valuer before making any purchase decision."
        )

    # ── Intent: flood / monsoon / water risk ──────────────────────────────────
    elif any(w in q for w in [
        "flood", "monsoon", "waterlog", "water risk", "drainage", "rain",
        "inundation", "cyclone"
    ]):
        monsoon_cards = _format_advice_cards(_advice_for(d["advice"], "Monsoon"))
        water_cards   = _format_advice_cards(_advice_for(d["advice"], "Water"))
        existing_advice = "\n".join(filter(None, [monsoon_cards, water_cards]))

        return (
            f"**Monsoon & Flood Risk for {d['locality']}, {d['city']}**\n\n"
            f"- **Flood risk:** {d['flood_risk']}\n"
            f"- **Water stress:** {d['water_stress']}\n"
            f"- **Water & Flood Resilience score:** {d['water_score']}/100\n"
            f"- **Annual precipitation:** {d['annual_precipitation_mm']} mm/year\n\n"
            + (f"**PropertyGuard advice for this property:**\n{existing_advice}\n\n"
               if existing_advice else "")
            + "**Pre-purchase checks:**\n"
            "1. Ask the seller if ground-floor or basement areas have ever been waterlogged.\n"
            "2. Verify the building's plinth/stilt elevation relative to surrounding road level.\n"
            "3. Inspect the proximity and clearance of storm-water drains around the building.\n"
            "4. Check the **Buyer Property Advice** tab — cards marked [WARNING] or [URGENT] "
            "flag direct risks requiring verification."
        )

    # ── Intent: air quality / pollution ──────────────────────────────────────
    elif any(w in q for w in [
        "air", "aqi", "pollution", "quality", "health", "breathe",
        "smog", "pm2", "dust"
    ]):
        aqi_cards = _format_advice_cards(_advice_for(d["advice"], "Air"))
        return (
            f"**Air Quality for {d['city']} — AQI {d['aqi']} ({d['air_quality_category']})**\n\n"
            f"- **Air Quality score:** {d['aqi_score']}/100\n\n"
            + (f"**PropertyGuard recommendation for this property:**\n{aqi_cards}\n\n"
               if aqi_cards else "")
            + "**Pre-purchase checks:**\n"
            "1. Visit the property on a weekday morning to assess traffic-related air quality.\n"
            "2. Check the floor level — higher floors typically see better ventilation.\n"
            "3. Review the **Buyer Property Advice** tab for specific guidance on air quality "
            "for this location."
        )

    # ── Intent: solar / energy ────────────────────────────────────────────────
    elif any(w in q for w in [
        "solar", "energy", "electricity", "rooftop", "saving", "panel",
        "kwh", "net meter", "renewable"
    ]):
        solar_cards = _format_advice_cards(_advice_for(d["advice"], "Solar"))
        return (
            f"**Solar Energy Potential for {d['city']} — Score {d['solar_score']}/100**\n\n"
            f"- **Solar insolation:** {d['solar_insolation']} kWh/m²/day (NASA POWER data)\n\n"
            + (f"**PropertyGuard recommendation for this property:**\n{solar_cards}\n\n"
               if solar_cards else "")
            + "**Before buying, confirm:**\n"
            "1. Are rooftop/terrace rights included in the sale deed?\n"
            "2. Does the housing society permit solar panel installation?\n"
            "3. Has the local DISCOM enabled net-metering for this building?\n"
            "4. Is the roof structurally suitable for panel mounting?"
        )

    # ── Intent: heat / thermal comfort ───────────────────────────────────────
    elif any(w in q for w in [
        "heat", "temperature", "hot", "cool", "thermal", "ac",
        "insulation", "summer", "comfort"
    ]):
        thermal_cards = _format_advice_cards(_advice_for(d["advice"], "Thermal"))
        return (
            f"**Thermal Comfort for {d['city']} — Score {d['thermal_score']}/100**\n\n"
            f"- **Peak summer temperature:** {d['summer_max_temp_c']}°C\n"
            f"- **Average humidity:** {d['humidity_pct']}%\n\n"
            + (f"**PropertyGuard recommendation for this property:**\n{thermal_cards}\n\n"
               if thermal_cards else "")
            + "**Pre-purchase checks:**\n"
            "1. Note which direction the main living areas and bedrooms face "
            "(west-facing rooms receive maximum afternoon heat).\n"
            "2. Ask whether the building has roof insulation or heat-reflective treatment.\n"
            "3. Check the **Buyer Property Advice** tab for thermal comfort guidance "
            "specific to this city."
        )

    # ── Intent: seller questions / pre-purchase checklist ────────────────────
    elif any(w in q for w in [
        "ask", "seller", "builder", "question", "checklist",
        "check", "buy", "purchase", "before buying", "due diligence"
    ]):
        urgent_cards = _format_advice_cards([
            a for a in d["advice"]
            if a["status"].lower() in ("urgent", "warning", "caution")
        ])
        return (
            f"**Key Questions to Ask the Seller / Builder — "
            f"{d['locality']}, {d['city']}**\n\n"
            "1. What is the RERA registration number for this project or property?\n"
            "2. Has the building experienced basement or ground-floor waterlogging "
            "in any monsoon season?\n"
            "3. Are rooftop solar panels and net-metering permitted by the housing society?\n"
            "4. What is the source of daily water supply — municipal mains or tanker-dependent?\n"
            "5. Is there a functional Rainwater Harvesting (RWH) system with recharge pits?\n"
            "6. What is the backup power capacity and does it cover lifts and all floors?\n"
            "7. What is the monthly maintenance charge and what services does it cover?\n"
            "8. Are there any pending legal disputes on the property title or land use?\n\n"
            + (
                f"**PropertyGuard has flagged these specific concerns for this property:**\n"
                f"{urgent_cards}"
                if urgent_cards else
                "PropertyGuard has not flagged any urgent concerns for this property. "
                "See the **Buyer Property Advice** tab for the full set of recommendations."
            )
        )

    # ── Intent: improve / upgrade sustainability ──────────────────────────────
    elif any(w in q for w in [
        "improve", "better", "upgrade", "green", "sustainable",
        "eco", "what can i do", "increase score", "enhance"
    ]):
        all_advice = _format_advice_cards(d["advice"])
        return (
            f"**How to Improve This Property's Sustainability — "
            f"{d['locality']}, {d['city']}**\n\n"
            "The best improvements depend on this property's weakest scoring pillars. "
            f"The current overall score is **{d['score']}/100 ({d['score_label']})**.\n\n"
            "**PropertyGuard's existing recommendations for this property:**\n"
            f"{all_advice}\n\n"
            "Review the **Buyer Property Advice** tab for the complete set of "
            "prioritised, property-specific recommendations."
        )

    # ── Intent: plain-language summary / overview ─────────────────────────────
    elif any(w in q for w in [
        "simple", "explain", "summary", "overview", "plain",
        "layman", "tell me about", "what is this", "overall"
    ]):
        benchmark_note = (
            d['benchmark_text'].split("|")[-1].strip()
            if d["loc_data"] else
            "No locality benchmark is available for direct comparison."
        )
        return (
            f"**Property Analysis Summary — {d['locality']}, {d['city']}**\n\n"
            f"**Price estimate:** PropertyGuard's ML model predicts this "
            f"{d['bedrooms']} BHK ({d['area_sqft']:,} sqft) property at "
            f"**{d['price_fmt']}** (±10% fair value range: "
            f"₹{d['min_est_l']:.1f}L – ₹{d['max_est_l']:.1f}L). "
            f"{benchmark_note}\n\n"
            f"**Climate & livability:** The Sustainability & Living Score is "
            f"**{d['score']}/100 ({d['score_label']})**. "
            "This reflects long-term livability across solar energy access "
            f"({d['solar_score']}/100), summer heat comfort ({d['thermal_score']}/100), "
            f"air quality ({d['aqi_score']}/100), and water/flood resilience "
            f"({d['water_score']}/100).\n\n"
            "**Next steps:** Review the **Buyer Property Advice** tab for specific "
            "pre-purchase action items, and the **NASA Climate & Sustainability** tab "
            "for the full climate breakdown."
        )

    # ── Default: unrecognised intent ──────────────────────────────────────────
    else:
        return (
            "**PropertyGuard Assistant** 🏠\n\n"
            "I can answer questions about this specific property's analysis. Try asking:\n\n"
            + "\n".join(f'- *"{sq}"*' for sq in SUGGESTED_QUESTIONS)
            + "\n\nAll answers are grounded in the ML predictions and NASA climate data "
            "already calculated for this property."
        )


def get_chat_response(
    user_question: str,
    chat_history: list,
    city: str,
    locality: str,
    bedrooms: int,
    area_sqft: int,
    is_resale: str,
    amenities_dict: dict,
    pred_price: float,
    loc_data: Optional[dict],
    climate_info: dict,
) -> str:
    """
    Public entry point called from app.py.

    1. Builds the property context from existing PropertyGuard outputs.
    2. Calls call_llm() — uses that response if an LLM is connected.
    3. Falls back to _rule_based_response() when call_llm() returns None.
    """
    context_string, structured_data = build_property_context(
        city=city,
        locality=locality,
        bedrooms=bedrooms,
        area_sqft=area_sqft,
        is_resale=is_resale,
        amenities_dict=amenities_dict,
        pred_price=pred_price,
        loc_data=loc_data,
        climate_info=climate_info,
    )

    llm_response = call_llm(SYSTEM_PROMPT, context_string, user_question, chat_history)
    if llm_response:
        return llm_response

    return _rule_based_response(user_question, structured_data)
