"""
Demonstration of FHR Analysis with Bradycardia/Tachycardia Detection
"""

# FHR Reference Ranges by Gestation Period
print("=" * 70)
print("FETAL HEART RATE (FHR) ANALYSIS DEMONSTRATION")
print("=" * 70)

print("\n📊 Normal FHR Ranges by Gestation Period:")
print("-" * 70)
ranges = [
    ("5-7 weeks", "90-130 bpm"),
    ("8-10 weeks", "170-180 bpm"),
    ("11-13 weeks", "140-160 bpm"),
    ("14-27 weeks", "120-160 bpm"),
    ("28-32 weeks", "110-160 bpm"),
    ("33-40 weeks", "110-150 bpm"),
]
for period, range_val in ranges:
    print(f"  {period:15s} → {range_val}")

print("\n\n📋 Example Analysis Scenarios:")
print("-" * 70)

scenarios = [
    {
        "gestation": "24 weeks",
        "fhr": 145,
        "expected_status": "✅ NORMAL",
        "description": "Heart rate within normal range (120-160 bpm)"
    },
    {
        "gestation": "24 weeks",
        "fhr": 100,
        "expected_status": "⚠️ BRADYCARDIA",
        "description": "Heart rate below normal range (< 120 bpm)"
    },
    {
        "gestation": "24 weeks",
        "fhr": 180,
        "expected_status": "⚠️ TACHYCARDIA",
        "description": "Heart rate above normal range (> 160 bpm)"
    },
    {
        "gestation": "8 weeks",
        "fhr": 175,
        "expected_status": "✅ NORMAL",
        "description": "Early pregnancy - normal range is higher (170-180 bpm)"
    },
    {
        "gestation": "35 weeks",
        "fhr": 130,
        "expected_status": "✅ NORMAL",
        "description": "Late pregnancy - heart rate typically lower (110-150 bpm)"
    },
    {
        "gestation": "30 weeks",
        "fhr": 95,
        "expected_status": "⚠️ BRADYCARDIA (SEVERE)",
        "description": "Significantly below normal - immediate medical attention needed"
    },
]

for i, scenario in enumerate(scenarios, 1):
    print(f"\nScenario {i}:")
    print(f"  Gestation Period: {scenario['gestation']}")
    print(f"  Measured FHR: {scenario['fhr']} bpm")
    print(f"  Status: {scenario['expected_status']}")
    print(f"  Analysis: {scenario['description']}")

print("\n\n🏥 API Response Structure:")
print("-" * 70)
print("""
When you call /analyze endpoint with gestation period, you'll receive:

{
  "predicted_label": "Normal",
  "confidence": 0.95,
  "gestation_period": "24 weeks",
  
  "heart_rate": {
    "average_fhr": 145.2,
    "beat_count": 24,
    "mean_fhr": 145.0,
    "median_fhr": 146.0,
    ...
  },
  
  "fhr_analysis": {
    "gestation_weeks": 24,
    "normal_range_min": 120,
    "normal_range_max": 160,
    "measured_fhr": 145.2,
    "fhr_status": "normal",
    "fhr_classification": "Normal Heart Rate",
    "severity": "none",
    "deviation": 0,
    "medical_concern": "No",
    "recommendation": "Heart rate is within normal range for gestational age."
  }
}

For BRADYCARDIA (example: FHR = 100 bpm at 24 weeks):
{
  "fhr_analysis": {
    "fhr_status": "bradycardia",
    "fhr_classification": "Bradycardia (Low Heart Rate)",
    "severity": "moderate",
    "deviation": -20,
    "medical_concern": "Yes - Heart rate below normal range",
    "recommendation": "Immediate medical consultation recommended. 
                       Bradycardia may indicate fetal distress."
  }
}

For TACHYCARDIA (example: FHR = 180 bpm at 24 weeks):
{
  "fhr_analysis": {
    "fhr_status": "tachycardia",
    "fhr_classification": "Tachycardia (High Heart Rate)",
    "severity": "moderate",
    "deviation": +20,
    "medical_concern": "Yes - Heart rate above normal range",
    "recommendation": "Medical consultation recommended. 
                       Tachycardia may indicate fetal stress or maternal fever."
  }
}
""")

print("\n" + "=" * 70)
print("✅ FHR Analysis with Bradycardia/Tachycardia Detection Ready!")
print("=" * 70)
