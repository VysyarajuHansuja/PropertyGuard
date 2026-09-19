"""
NASA Climate & Sustainability Engine
------------------------------------
Provides microclimate data lookup, environmental risk metrics,
Sustainability & Living Score (0-100), and actionable property advice
using NASA POWER (Prediction Of Worldwide Energy Resources) climate indicators.
"""

import json
import os

CITY_CLIMATE_BASE = {
    "Bengaluru": {
        "lat": 12.9716,
        "lon": 77.5946,
        "max_summer_temp_c": 33.8,
        "avg_annual_temp_c": 24.1,
        "solar_insolation_kwh_m2_day": 5.65,
        "annual_precipitation_mm": 970,
        "monsoon_flood_risk": "Moderate",
        "avg_aqi": 68,
        "air_quality_category": "Moderate",
        "avg_relative_humidity_pct": 62,
        "water_stress_index": "High (Groundwater Depleted)",
        "green_cover_pct": 21.5
    },
    "Chennai": {
        "lat": 13.0827,
        "lon": 80.2707,
        "max_summer_temp_c": 38.6,
        "avg_annual_temp_c": 28.6,
        "solar_insolation_kwh_m2_day": 5.42,
        "annual_precipitation_mm": 1420,
        "monsoon_flood_risk": "High (Northeast Monsoon Inundation)",
        "avg_aqi": 78,
        "air_quality_category": "Moderate",
        "avg_relative_humidity_pct": 74,
        "water_stress_index": "Severe (Coastal Salinity / Tanker Dependent)",
        "green_cover_pct": 14.8
    },
    "Delhi": {
        "lat": 28.6139,
        "lon": 77.2090,
        "max_summer_temp_c": 42.4,
        "avg_annual_temp_c": 25.2,
        "solar_insolation_kwh_m2_day": 5.25,
        "annual_precipitation_mm": 740,
        "monsoon_flood_risk": "Moderate (Yamuna Floodplain Risk)",
        "avg_aqi": 245,
        "air_quality_category": "Poor to Severe",
        "avg_relative_humidity_pct": 53,
        "water_stress_index": "High",
        "green_cover_pct": 23.1
    },
    "Hyderabad": {
        "lat": 17.3850,
        "lon": 78.4867,
        "max_summer_temp_c": 39.5,
        "avg_annual_temp_c": 26.6,
        "solar_insolation_kwh_m2_day": 5.58,
        "annual_precipitation_mm": 840,
        "monsoon_flood_risk": "Moderate",
        "avg_aqi": 94,
        "air_quality_category": "Moderate",
        "avg_relative_humidity_pct": 58,
        "water_stress_index": "Moderate",
        "green_cover_pct": 17.2
    },
    "Kolkata": {
        "lat": 22.5726,
        "lon": 88.3639,
        "max_summer_temp_c": 37.8,
        "avg_annual_temp_c": 26.8,
        "solar_insolation_kwh_m2_day": 4.85,
        "annual_precipitation_mm": 1750,
        "monsoon_flood_risk": "High (Cyclone & Drainage Waterlogging)",
        "avg_aqi": 165,
        "air_quality_category": "Unhealthy",
        "avg_relative_humidity_pct": 76,
        "water_stress_index": "Moderate",
        "green_cover_pct": 12.4
    },
    "Mumbai": {
        "lat": 19.0760,
        "lon": 72.8777,
        "max_summer_temp_c": 35.2,
        "avg_annual_temp_c": 27.4,
        "solar_insolation_kwh_m2_day": 5.15,
        "annual_precipitation_mm": 2420,
        "monsoon_flood_risk": "Very High (Heavy Monsoon Waterlogging)",
        "avg_aqi": 115,
        "air_quality_category": "Moderate",
        "avg_relative_humidity_pct": 78,
        "water_stress_index": "Low (Lake Reservoir Fed)",
        "green_cover_pct": 18.2
    }
}

LOCALITY_CLIMATE_MODIFIERS = {
    # High-density / Heat Island / Flood Vulnerable micro-localities
    "Kurla": {"temp_mod": 1.5, "flood_mod": "Extreme", "aqi_mod": 15},
    "Dharavi": {"temp_mod": 2.0, "flood_mod": "Extreme", "aqi_mod": 20},
    "Velachery": {"temp_mod": 0.5, "flood_mod": "Extreme", "aqi_mod": 5},
    "Anand Vihar": {"temp_mod": 1.8, "flood_mod": "Moderate", "aqi_mod": 60},
    "Whitefield": {"temp_mod": -0.5, "flood_mod": "Low", "aqi_mod": -10},
    "Gachibowli": {"temp_mod": -0.8, "flood_mod": "Low", "aqi_mod": -12},
    "New Town": {"temp_mod": 0.0, "flood_mod": "Moderate", "aqi_mod": -5},
    "Bandra West": {"temp_mod": 0.2, "flood_mod": "Moderate", "aqi_mod": 0},
    "Indiranagar": {"temp_mod": -0.6, "flood_mod": "Low", "aqi_mod": -8}
}

def get_climate_data(city: str, locality: str = "") -> dict:
    """
    Lookup NASA climate profile for city & locality,
    and compute Sustainability Score and Actionable Buyer Advice.
    """
    base = CITY_CLIMATE_BASE.get(city, CITY_CLIMATE_BASE["Bengaluru"]).copy()
    
    # Apply locality micro-modifier if present
    mod = LOCALITY_CLIMATE_MODIFIERS.get(locality, {"temp_mod": 0.0, "flood_mod": None, "aqi_mod": 0})
    
    summer_temp = round(base["max_summer_temp_c"] + mod["temp_mod"], 1)
    aqi = base["avg_aqi"] + mod["aqi_mod"]
    flood_risk = mod["flood_mod"] if mod["flood_mod"] else base["monsoon_flood_risk"]
    
    # Calculate Sub-Scores (0 to 100)
    # Solar Score: 5.5+ kWh/m2/day is 100
    solar_score = min(100, max(40, int((base["solar_insolation_kwh_m2_day"] / 6.0) * 100)))
    
    # Thermal Comfort Score: 30C is 100, 45C is 20
    thermal_score = min(100, max(20, int(100 - (summer_temp - 30.0) * 5.0)))
    
    # Air Quality Score: AQI 30 is 100, AQI 300 is 10
    aqi_score = min(100, max(10, int(100 - (aqi / 300.0) * 85)))
    
    # Water & Monsoon Risk Score
    flood_penalty = {"Low": 5, "Moderate": 20, "High": 35, "Very High": 45, "Extreme": 55}.get(flood_risk.split()[0], 20)
    water_score = max(20, 100 - flood_penalty)
    
    # Total Sustainability & Living Score
    overall_sustainability_score = int(
        (solar_score * 0.25) +
        (thermal_score * 0.25) +
        (aqi_score * 0.25) +
        (water_score * 0.25)
    )
    
    # Generate Buyer Living & Property Advice
    advice_list = []
    
    # 1. Solar Rooftop Advice
    est_annual_solar_kwh = int(base["solar_insolation_kwh_m2_day"] * 365 * 0.75 * 4) # for 4kW system
    annual_savings_inr = est_annual_solar_kwh * 7.5
    advice_list.append({
        "category": "Solar Energy & Electricity",
        "title": "Solar Rooftop High Potential",
        "score_impact": f"{solar_score}/100",
        "status": "Recommended",
        "icon": "☀️",
        "description": f"Installing a 4kW solar rooftop system can generate ~{est_annual_solar_kwh:,} kWh/year, saving approximately ₹{annual_savings_inr:,.0f} annually on electricity bills."
    })
    
    # 2. Water & Rainwater Harvesting Advice
    if "High" in base["water_stress_index"] or "Severe" in base["water_stress_index"]:
        advice_list.append({
            "category": "Water Security",
            "title": "Rainwater Harvesting & Filtration Essential",
            "score_impact": f"{water_score}/100",
            "status": "Urgent",
            "icon": "🌧️",
            "description": f"Locality has {base['water_stress_index'].lower()}. Check if property features active Rainwater Harvesting (RWH) and groundwater recharge pits."
        })
    else:
        advice_list.append({
            "category": "Water Security",
            "title": "Good Water Security",
            "score_impact": f"{water_score}/100",
            "status": "Good",
            "icon": "💧",
            "description": "Locality has stable municipal water supply. Standard dual-plumbing is recommended for greywater reuse."
        })
        
    # 3. Heat & Insulation Advice
    if summer_temp >= 38.0:
        advice_list.append({
            "category": "Thermal Comfort",
            "title": "Cool Roof Paint & Double Glazing Advised",
            "score_impact": f"{thermal_score}/100",
            "status": "Recommended",
            "icon": "🌡️",
            "description": f"Summer peak surface temps reach {summer_temp}°C. High-SRI cool roof paint and double-glazed windows will reduce AC cooling loads by up to 25%."
        })
    else:
        advice_list.append({
            "category": "Thermal Comfort",
            "title": "Moderate Summer Thermal Load",
            "score_impact": f"{thermal_score}/100",
            "status": "Good",
            "icon": "🍃",
            "description": f"Peak summer temperatures (~{summer_temp}°C) are manageable with cross-ventilation and 5-star inverter ACs."
        })

    # 4. Air Quality & Health Advice
    if aqi >= 150:
        advice_list.append({
            "category": "Air Quality & Health",
            "title": "HEPA Air Purifier & Sealed Ventilation Recommended",
            "score_impact": f"{aqi_score}/100",
            "status": "Caution",
            "icon": "😷",
            "description": f"Average AQI is {aqi} ({base['air_quality_category']}). Indoor HEPA air filtration and indoor green plants (Snake plant, Areca palm) are strongly advised."
        })
    else:
        advice_list.append({
            "category": "Air Quality & Health",
            "title": "Clean to Moderate Air Quality",
            "score_impact": f"{aqi_score}/100",
            "status": "Good",
            "icon": "🌿",
            "description": f"Locality experiences clean to moderate air quality (AQI ~{aqi}). Good outdoor natural ventilation."
        })
        
    # 5. Flood Warning
    if "High" in flood_risk or "Extreme" in flood_risk:
        advice_list.append({
            "category": "Monsoon Risk",
            "title": "Monsoon Waterlogging & Drainage Caution",
            "score_impact": f"{water_score}/100",
            "status": "Warning",
            "icon": "🌊",
            "description": f"Monsoon Flood Vulnerability is {flood_risk}. Verify stilt level parking elevation, basement sump pumps, and storm-water drain clearance."
        })

    return {
        "city": city,
        "locality": locality if locality else city,
        "lat": base["lat"],
        "lon": base["lon"],
        "sustainability_score": overall_sustainability_score,
        "metrics": {
            "summer_max_temp_c": summer_temp,
            "solar_insolation": base["solar_insolation_kwh_m2_day"],
            "annual_precipitation_mm": base["annual_precipitation_mm"],
            "flood_risk": flood_risk,
            "aqi": aqi,
            "air_quality_category": base["air_quality_category"],
            "humidity_pct": base["avg_relative_humidity_pct"],
            "water_stress": base["water_stress_index"],
            "green_cover_pct": base["green_cover_pct"],
            "solar_score": solar_score,
            "thermal_score": thermal_score,
            "aqi_score": aqi_score,
            "water_score": water_score
        },
        "advice": advice_list
    }

if __name__ == "__main__":
    test_res = get_climate_data("Mumbai", "Kurla")
    print(json.dumps(test_res, indent=2))
