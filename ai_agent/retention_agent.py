import os
import json
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Strip quotes if present
GEMINI_API_KEY = GEMINI_API_KEY.strip('\'"')

# Flag to check if the key is valid and not a placeholder
IS_KEY_CONFIGURED = (
    GEMINI_API_KEY != "" 
    and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE" 
    and len(GEMINI_API_KEY) > 10
)

# Try validating Gemini API configuration
try:
    if IS_KEY_CONFIGURED:
        logger.info("Gemini API Key configured. Will use direct REST API integration.")
    else:
        logger.warning("Gemini API key is not configured. AI Agent will operate in Fallback/Mock mode.")
except Exception as e:
    logger.error(f"Error validating Gemini API configuration: {e}. AI Agent will run in Fallback/Mock mode.")
    IS_KEY_CONFIGURED = False

def get_fallback_plan(profile, risk_score):
    """
    Generates a high-quality, customized, rule-based retention plan when Gemini is unavailable.
    Uses customer details to create logical strategies and a beautifully written email.
    """
    tenure = profile.get("tenure", 1)
    monthly_charges = profile.get("MonthlyCharges", 50.0)
    contract = profile.get("Contract", "Month-to-month")
    internet = profile.get("InternetService", "DSL")
    tech_support = profile.get("TechSupport", "No")
    payment_method = profile.get("PaymentMethod", "Electronic check")
    
    reasons = []
    strategies = []
    offers = []
    
    # 1. Custom Churn Reasons based on actual customer profile
    if contract == "Month-to-month":
        reasons.append("Flexible month-to-month billing creates low switching barriers and high susceptibility to competitor offers.")
    if monthly_charges > 75.0:
        reasons.append(f"Elevated monthly charges of ${monthly_charges:.2f} exceed comfortable budget thresholds, driving price-sensitivity.")
    if internet == "Fiber optic":
        reasons.append("Fiber optic subscription provides premium speed but has historically higher billing rates and service complaint frequencies.")
    if tech_support == "No":
        reasons.append("Absence of specialized Tech Support leaves the customer vulnerable to technical frustrations, reducing brand reliability.")
    if tenure <= 12:
        reasons.append(f"Short customer tenure ({tenure} months) reflects a lack of established brand loyalty and higher risk during onboarding.")
    if payment_method == "Electronic check":
        reasons.append("Manual 'Electronic check' payment requires recurring monthly effort, increasing opportunities to reconsider the subscription.")
        
    # Standard catch-all if profile has none of the above
    if not reasons:
        reasons.append("Sub-optimal digital engagement and lack of multi-service bundling options.")
        reasons.append("Risk of competitor migration due to generic non-contract billing status.")

    # 2. Specific Actionable Strategies
    if contract == "Month-to-month":
        strategies.append("Pitch a 1-year annual contract transition securing a monthly rate discount, adding immediate contract stability.")
    else:
        strategies.append("Propose a premium service upgrade with a loyalty waiver to lock in another cycle of satisfaction.")
        
    if monthly_charges > 70.0:
        strategies.append("Conduct a proactive tariff audit to migrate the customer to a modern bundle with identical speed but optimized costs.")
    else:
        strategies.append("Bundle a complementary value-add service (e.g., streaming or cloud storage) to increase product attachment.")
        
    if tech_support == "No":
        strategies.append("Offer a 6-month trial of 24/7 Priority Tech Support to directly address potential system issues.")
    else:
        strategies.append("Schedule a dedicated customer relationship check-in to verify line stability and equipment performance.")
        
    strategies.append("Propose transitioning from manual invoicing to Automatic Credit Card billing in exchange for a one-time bill credit.")

    # 3. Targeted Offers
    if contract == "Month-to-month":
        offers.append("1-Year Loyalty Contract: 15% discount on the base subscription fee for 12 months with price-lock guarantee.")
    else:
        offers.append("Loyalty Premium Waiver: Free speed boost or additional service tier for 6 months.")
        
    if tech_support == "No" or internet == "Fiber optic":
        offers.append("Elite Care Package: Free 6 months of VIP technical support and free modem/router hardware upgrades.")
    else:
        offers.append("Autopay Bonus: $10 bill credit upon signing up for paperless Autopay, combined with a 5% monthly billing discount.")

    # 4. Professional Retention Email Draft
    email_subject = f"Exclusive Loyalty Rewards: We Value Having You With Us!"
    
    email_body = f"""Dear Valued Customer,

Thank you for choosing us as your trusted service provider. We are constantly monitoring our systems to ensure we are delivering the highest possible value, performance, and reliability to our long-standing users.

We recently reviewed your account profile (Tenure: {tenure} months) and want to proactively thank you for your business. We want to make sure your service matches your needs and fits your budget. We notice you are currently enjoying our {internet} internet service at ${monthly_charges:.2f} per month. 

To show our appreciation for your loyalty, our customer success team has authorized some exclusive rewards specifically for your account:

1. {offers[0]}
2. {offers[1] if len(offers) > 1 else 'Free 3 Months of Premium Tech Support and Security suite.'}

Making sure you have a seamless experience is our number one priority. If you would like to claim these rewards or review your current plan options, please simply reply to this email, or call our Dedicated Loyalty Hot-line directly at 1-800-LOYALTY.

Thank you once again for your business and partnership.

Warm regards,

Sarah Jenkins
Senior Customer Success Manager
Global Telecom Solutions"""

    return {
        "reasons": reasons[:3],
        "strategies": strategies[:3],
        "offers": offers[:2],
        "email_subject": email_subject,
        "email_body": email_body,
        "method": "Rule-Based Heuristics (AI Key Missing)"
    }

def generate_retention_plan(customer_profile, risk_score):
    """
    Generates a personalized customer retention plan using Gemini API (or the high-quality fallback).
    """
    if not IS_KEY_CONFIGURED:
        return get_fallback_plan(customer_profile, risk_score)
        
    try:
        # Construct highly contextual prompt
        prompt = f"""
        You are an elite Customer Retention Specialist for a major telecommunications provider.
        Analyze this high-risk customer profile:
        - Tenure with company: {customer_profile.get('tenure', 'N/A')} months
        - Contract Type: {customer_profile.get('Contract', 'N/A')}
        - Monthly Bill Charges: ${customer_profile.get('MonthlyCharges', 'N/A')}
        - Total Services Subscribed: {customer_profile.get('total_services', 'N/A')} active services
        - Internet Infrastructure: {customer_profile.get('InternetService', 'N/A')}
        - Has Tech Support: {customer_profile.get('TechSupport', 'N/A')}
        - Payment Method: {customer_profile.get('PaymentMethod', 'N/A')}
        - Paperless Billing: {customer_profile.get('PaperlessBilling', 'N/A')}
        
        Calculated Customer Churn Risk: {risk_score:.1f}%
        
        Based on this specific data:
        1. Explain exactly 3 key technical reasons why this customer is likely to churn (e.g., billing, contract type, lack of features, etc.). Keep explanations precise and data-driven.
        2. Suggest 3 specific, actionable retention strategies for our account managers (e.g., tariff optimization, billing upgrades, support bundles).
        3. Recommend 2 compelling and targeted loyalty offers we should extend to this customer (e.g., 20% discount, free tech support trial, price locks).
        4. Generate a highly personalized, empathetic, and professional retention email from "Sarah Jenkins, Customer Success Manager" to the customer. Use the customer details naturally.
        
        You MUST respond in strict JSON format. Do not include any markdown wrappers (like ```json). Return ONLY the raw JSON string.
        The JSON structure MUST match this exactly:
        {{
          "reasons": [
            "Reason 1 using data",
            "Reason 2 using data",
            "Reason 3 using data"
          ],
          "strategies": [
            "Actionable strategy 1",
            "Actionable strategy 2",
            "Actionable strategy 3"
          ],
          "offers": [
            "Specific offer 1",
            "Specific offer 2"
          ],
          "email_subject": "A compelling, positive email subject line",
          "email_body": "Full body of the email starting with a polite greeting, referencing their specific tenure and monthly charges, clearly laying out the loyalty offers, and concluding with a warm signature."
        }}
        """
        
        # Request generation via REST API (zero-dependency for Python 3.8 compatibility)
        import urllib.request
        import urllib.error
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
        headers = {'Content-Type': 'application/json'}
        data = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }
        req_data = json.dumps(data).encode('utf-8')
        req = urllib.request.Request(url, data=req_data, headers=headers, method='POST')
        
        with urllib.request.urlopen(req, timeout=20) as response:
            res_body = response.read().decode('utf-8')
            res_json = json.loads(res_body)
            text = res_json['candidates'][0]['content']['parts'][0]['text'].strip()
        
        # Clean JSON if model wrapped it in markdown codeblocks
        if text.startswith("```"):
            # Remove leading ```json or ```
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
            
        plan = json.loads(text)
        plan["method"] = "Google Gemini AI (REST Active)"
        
        return plan
        
    except Exception as e:
        logger.error(f"Gemini REST API request failed: {e}. Falling back to Rule-Based plan.")
        return get_fallback_plan(customer_profile, risk_score)

if __name__ == "__main__":
    # Test script
    dummy_profile = {
        'tenure': 4,
        'Contract': 'Month-to-month',
        'MonthlyCharges': 85.50,
        'InternetService': 'Fiber optic',
        'TechSupport': 'No',
        'PaymentMethod': 'Electronic check',
        'total_services': 2
    }
    
    plan = generate_retention_plan(dummy_profile, 88.5)
    print(f"Generated Plan Method: {plan['method']}")
    print(f"Reasons: {plan['reasons']}")
    print(f"Offers: {plan['offers']}")
    print(f"Email Subject: {plan['email_subject']}")
