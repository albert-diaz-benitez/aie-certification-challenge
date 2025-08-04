"""
Defines the golden dataset for evaluating the RAG system using RAGAS.

This module contains real-world and synthetic insurance emails with ground truth policy information
for evaluating extraction accuracy.
"""

from typing import Any, Dict, List

# Golden dataset: collection of sample emails with ground truth policy information
# Each entry contains the email content and the correct policy field values
GOLDEN_DATASET = [
    {
        "email_id": "test_email_001",
        "subject": "New Policy Details - Johnson Manufacturing",
        "sender": "underwriter@insuranceco.com",
        "body_text": """Dear Agent,

We're pleased to inform you that the policy for Johnson Manufacturing has been approved.
Please find the details below:

Policy Insured: Johnson Manufacturing Inc.
Line of Business: Property
Effective Date: 2025-09-01
Expected Inception Date: 2025-09-15
Target Premium: $45,750.00

Please review the details and let us know if you have any questions.

Best regards,
Sarah Williams
Senior Underwriter
Insurance Co.
""",
        "ground_truth": {
            "policy_insured": "Johnson Manufacturing Inc.",
            "line_of_business": "Property",
            "effective_date": "2025-09-01",
            "expected_inception_date": "2025-09-15",
            "target_premium": "$45,750.00",
        },
    },
    {
        "email_id": "test_email_002",
        "subject": "RE: Policy Quote for Riverside Hospital",
        "sender": "quotes@abcinsurance.com",
        "body_text": """Hello Tom,

As per our discussion yesterday, I'm sending over the updated quote for Riverside Hospital.

The policy details are as follows:
- Insured Party: Riverside Community Hospital
- Coverage Type: Medical Liability
- Premium: $128,500
- Policy Start: October 1, 2025
- Expected Implementation: September 20, 2025

Note that this is a 15% increase from last year due to the new wing addition.

Please confirm if you would like to proceed.

Regards,
Michael Chen
ABC Insurance
""",
        "ground_truth": {
            "policy_insured": "Riverside Community Hospital",
            "line_of_business": "Medical Liability",
            "effective_date": "2025-10-01",
            "expected_inception_date": "2025-09-20",
            "target_premium": "$128,500",
        },
    },
    {
        "email_id": "test_email_003",
        "subject": "Smith Auto Shop - Policy Renewal Information",
        "sender": "renewals@insuranceproviders.com",
        "body_text": """Policy Renewal Notification

Customer: Smith Auto Repair Shop
Policy Type: Commercial Auto
Renewal Date: November 15, 2025
Implementation Timeline: We recommend beginning the renewal process by November 1, 2025
Annual Premium: $8,250

Additional Notes:
- Fleet size increased from 5 to 7 vehicles
- Added coverage for new diagnostic equipment
- Maintaining same deductible as previous policy ($1,000)

Please contact your account manager if you wish to make any changes to the policy before renewal.

Insurance Providers Inc.
""",
        "ground_truth": {
            "policy_insured": "Smith Auto Repair Shop",
            "line_of_business": "Commercial Auto",
            "effective_date": "2025-11-15",
            "expected_inception_date": "2025-11-01",
            "target_premium": "$8,250",
        },
    },
    {
        "email_id": "test_email_004",
        "subject": "New Policy Quote - Green Valley Apartments",
        "sender": "commercial@insurancesolutions.com",
        "body_text": """Dear Ms. Rodriguez,

Thank you for submitting the application for Green Valley Apartment Complex. Based on the information provided, we've prepared the following quote:

Property: Green Valley Apartments LLC
Insurance Class: Commercial Property
Coverage Begins: January 10, 2026
Quote Valid Until: December 15, 2025
Annual Cost: $67,890

Coverage includes:
- Property damage
- Liability ($2M)
- Loss of income
- Natural disaster coverage (excluding flood)

The underwriter notes that installing the recommended security system could reduce the premium by 8%.

Please let me know if you'd like to proceed with this quote.

Best,
David Thompson
Commercial Insurance Solutions
""",
        "ground_truth": {
            "policy_insured": "Green Valley Apartments LLC",
            "line_of_business": "Commercial Property",
            "effective_date": "2026-01-10",
            "expected_inception_date": "2025-12-15",
            "target_premium": "$67,890",
        },
    },
    {
        "email_id": "test_email_005",
        "subject": "Policy Information for Downtown Bistro",
        "sender": "small.business@insuranceco.com",
        "body_text": """Hello Mark,

I'm writing to confirm the details of the policy we discussed for Downtown Bistro.

The details are:
- Business Name: Downtown Bistro & Catering
- Policy: Restaurant & Catering Insurance
- Policy Effective: 2025-10-15
- Implementation Date: 2025-10-01
- Premium: $12,450 annually (can be paid in quarterly installments of $3,245)

This policy includes coverage for:
- General liability
- Property damage
- Food contamination
- Liquor liability
- Workers' compensation

Please sign the attached documents and return them by September 20th to ensure timely processing.

Thanks,
Lisa Wong
Small Business Insurance Specialist
""",
        "ground_truth": {
            "policy_insured": "Downtown Bistro & Catering",
            "line_of_business": "Restaurant & Catering Insurance",
            "effective_date": "2025-10-15",
            "expected_inception_date": "2025-10-01",
            "target_premium": "$12,450",
        },
    },
    {
        "email_id": "golden_email_001",
        "subject": "Submission for ABC Manufacturing Co. - Property Insurance",
        "sender": "broker@abcbrokerage.com",
        "body_text": """Dear Jamie Underwriter,

I hope you're well. ABC Manufacturing Co. (client) is seeking property insurance for their primary facility in Chicago, and we would like to submit this risk for your evaluation. ABC Manufacturing is a 20-year-old company specializing in industrial machinery production. They have been with their current carrier for 5+ years but are marketing their program this year due to capacity withdrawal by the incumbent. We believe this account is an excellent fit for your underwriting appetite given its strong risk controls and stable loss history.

Risk Overview: The insured's property is a single-location manufacturing plant/warehouse at 123 Industrial Way, Chicago. The building is 90,000 sq. ft., steel frame with masonry walls (Construction: Class JM), housing both production and storage. It was built in 2010 and is in excellent condition. The facility has modern safety protections – it's 100% sprinklered (wet system) with a central station fire alarm and security alarm. There is a 24/7 security guard service on-site. The location is in a moderate hazard area (light industrial park) with good fire brigade access (1 mile from fire station, hydrants on site).

Coverage Needs: The client requires a Commercial Property policy covering the building and contents. The total Insured Value for the building is $8,000,000 and contents (equipment and stock) $2,000,000 (TIV $10M). They also carry $5M in Business Interruption (12-month gross earnings). We are seeking "All Risk" special form coverage, including replacement cost valuation and a $50,000 deductible. Please include coverage for equipment breakdown. No flood or quake exposure at this location (not in a flood zone). The insured is focused on comprehensive coverage and is less price-sensitive; however, the target premium is around $60,000 annually based on expiring terms.

Loss History: Excellent loss record – no property claims in the last 5 years. (Prior to that, one minor $5,200 water leak claim occurred in 2018, which was fully remedied – upgraded a section of plumbing.) We have included 5-year loss run reports from the current carrier in the attachments for verification. The insured's proactive maintenance and risk management programs have clearly paid off with this clean loss history.

Risk Management: ABC has a dedicated Risk Manager on staff and a robust safety program. They conduct quarterly sprinkler system inspections and annual thermographic scans of electrical systems to prevent fire losses. Housekeeping is excellent and they maintain a detailed emergency response plan. These factors, along with management's commitment to safety, make the risk well-managed and attractive from an underwriter's perspective. We have also attached a brief Risk Profile Summary describing these controls in more detail.

Submission Documents: Please find attached the following documents for your review:
ACORD Property Application (completed ACORD 125 and 140 forms, signed)
Property Schedule of Values (Excel file with location details and values)
5-Year Loss Runs (PDF from current insurer showing no losses)
Risk Profile Summary (PDF narrative with photos and risk control details)

The insured's renewal date is 1 Oct 2025, so we need an indication or quote by Sept 15 if possible. (They require some lead time to review options internally.) Thank you for your consideration of this submission. We genuinely value our partnership with XYZ Underwriting and are excited for your evaluation of ABC Manufacturing Co. Please let me know if you need any additional information or have any questions – I am happy to assist promptly.

Sincerely,
Broker | Account Executive, Commercial Lines
ABC Insurance Brokerage
(555) 123-4567 | name@abcbrokerage.com""",
        "ground_truth": {
            "policy_insured": "ABC Manufacturing Co.",
            "line_of_business": "Commercial Property",
            "effective_date": "2025-10-01",
            "expected_inception_date": "2025-09-15",  # Using the quote date as inception
            "target_premium": "$60,000",
        },
    },
    {
        "email_id": "golden_email_002",
        "subject": "Coverage Analysis - North Shore Medical Group Professional Liability",
        "sender": "underwriter@medliability.com",
        "body_text": """COVERAGE ANALYSIS MEMORANDUM
DATE: August 2, 2025
TO: Risk Management Department
FROM: Medical Professional Liability Underwriting
RE: North Shore Medical Group - Professional Liability Coverage Analysis

This memorandum presents our preliminary coverage analysis for North Shore Medical Group's professional liability insurance program. Our team has completed a thorough evaluation of the submission materials and risk factors associated with this multi-specialty physician group.

APPLICANT OVERVIEW:
North Shore Medical Group is a multi-specialty physician group with 47 physicians across 5 locations in the greater Boston area. Their specialties include Internal Medicine (12), Family Practice (8), Pediatrics (6), Cardiology (5), Orthopedic Surgery (4), General Surgery (3), Obstetrics/Gynecology (3), Dermatology (3), Gastroenterology (2), and Neurology (1).

COVERAGE SPECIFICATIONS:
• Policy Type: Claims-Made Medical Professional Liability
• Retroactive Date: October 15, 2020 (maintaining existing)
• Proposed Effective Date: November 1, 2025
• Policy Period: 12 months
• Limits of Liability: $1M per claim / $3M aggregate
• Deductible: $25,000 per claim
• Defense Costs: Outside the limits
• Consent to Settle: Required (Hammer Clause included)
• Extended Reporting Period: Available, 12/24/36/60 month options

RISK ASSESSMENT:
North Shore Medical Group presents a moderate risk profile based on our evaluation metrics. The group maintains rigorous credentialing processes and has implemented a robust EHR system with integrated clinical decision support. Their risk management program includes regular peer review, patient satisfaction monitoring, and mandatory continuing education requirements exceeding state minimums.

CLAIMS HISTORY:
The group has experienced 3 claims over the past 5 years with a total incurred amount of $475,000. This represents a claims frequency below our portfolio average for similar-sized multi-specialty groups. The severity metrics are also favorable, with only one claim exceeding $250,000 (a $325,000 settlement involving a delayed diagnosis case).

PREMIUM INDICATION:
Based on our analysis, we are providing a premium indication of $387,500 for the specified coverage. This represents a 5% increase over expiring, reflecting moderate medical inflation and the addition of two new physicians since last renewal.

UNDERWRITING REQUIREMENTS:
1. Updated credentialing verification for all physicians
2. Completion of our supplemental risk assessment questionnaire
3. Implementation timeline for the planned telehealth program expansion
4. Verification of tail coverage for recently departed physicians

TIMELINE:
We would like to provide a formal quote by September 15, 2025, to allow adequate time before the November 1 effective date. Please submit all required documentation by September 1 to facilitate this timeline.

If you have any questions or require clarification on any aspect of this analysis, please contact our underwriting department.

Elizabeth Johnson
Senior Medical Professional Liability Underwriter
MedLiability Insurance Company""",
        "ground_truth": {
            "policy_insured": "North Shore Medical Group",
            "line_of_business": "Medical Professional Liability",
            "effective_date": "2025-11-01",
            "expected_inception_date": "2025-09-15",  # Using the quote date
            "target_premium": "$387,500",
        },
    },
    {
        "email_id": "golden_email_003",
        "subject": "Urgent: Harbor City Tower - Flood Policy Renewal",
        "sender": "specialtyrisks@floodinsurance.com",
        "body_text": """URGENT - POLICY RENEWAL NOTICE: ACTION REQUIRED
Harbor City Tower - Private Flood Insurance Program

To: Asset Management Team, Harbor City Properties LLC
From: Specialty Risks Underwriting Department
Date: August 1, 2025
Re: PRIVATE FLOOD INSURANCE RENEWAL - POLICY #FLD-8294761

RENEWAL DEADLINE: AUGUST 25, 2025

Dear Valued Client,

This communication serves as your official 30-day notice that your Private Flood Insurance Policy #FLD-8294761 for Harbor City Tower will expire on September 1, 2025. Due to the property's location in a Special Flood Hazard Area (Zone AE) and the critical nature of flood coverage for your mortgage compliance, we strongly recommend initiating the renewal process immediately.

PROPERTY INFORMATION:
Property Name: Harbor City Tower
Address: 1200 Harbor Boulevard, Miami, FL 33132
Building Occupancy: Mixed-Use (Commercial/Residential)
Construction: Reinforced Concrete, 32 stories
Year Built: 2018

CURRENT COVERAGE:
Building Coverage: $50,000,000
Contents Coverage: $2,500,000
Business Interruption: $5,000,000 (6-month period of restoration)
Deductible: $250,000 per occurrence
Current Premium: $875,000

RENEWAL CONSIDERATIONS:
Our catastrophe modeling team has completed updated flood risk projections based on the latest NOAA climate data. Your property's risk profile has been moderately impacted by these updates. Additionally, there have been significant changes in the reinsurance market for coastal flood exposures in South Florida.

RENEWAL OPTIONS:

OPTION 1 - MAINTAIN CURRENT COVERAGE
• Coverage limits remain unchanged
• Deductible: $250,000
• Premium: $962,500 (10% increase)

OPTION 2 - INCREASED DEDUCTIBLE
• Coverage limits remain unchanged
• Deductible: $500,000 (increased)
• Premium: $875,000 (no change from expiring)

OPTION 3 - LAYERED PROGRAM
• Primary Layer: $25M excess $250K deductible
• Excess Layer: $25M excess $25M
• Premium: $918,750 (5% increase)

LOSS CONTROL RECOMMENDATIONS:
Our engineering team's assessment from April 2025 identified several opportunities to mitigate flood exposure:
1. Installation of additional backflow prevention devices (Estimated cost: $45,000)
2. Reinforcement of lobby flood barriers (Estimated cost: $85,000)
3. Relocation of critical electrical infrastructure from basement to 3rd floor (Estimated cost: $275,000)

Implementation of these measures could qualify your property for our Resilience Credit Program, potentially reducing premiums by up to 7.5%.

NEXT STEPS:
1. Please indicate your preferred renewal option by August 15, 2025
2. Complete and sign the attached renewal application
3. Provide updated statement of values for contents if any changes have occurred
4. Submit documentation of any implemented flood mitigation measures

We value your business and are committed to providing comprehensive flood protection for Harbor City Tower. If you have any questions or wish to discuss these options in detail, please contact your Account Executive, Jessica Martinez, at (305) 555-7890 or jmartinez@floodinsurance.com.

Sincerely,

Robert Thompson
Senior Specialty Lines Underwriter
Flood Insurance Specialists, Inc.
FL License #P078542""",
        "ground_truth": {
            "policy_insured": "Harbor City Properties LLC",
            "line_of_business": "Private Flood Insurance",
            "effective_date": "2025-09-01",
            "expected_inception_date": "2025-08-25",  # Using the renewal deadline
            "target_premium": "$962,500",  # Using Option 1 as the default
        },
    },
]


# Generate question-answer pairs for each sample and field
def generate_qa_pairs() -> List[Dict[str, Any]]:
    """
    Generate question-answer pairs for RAGAS evaluation.

    Returns:
        List of dictionaries containing questions, answers, and contexts.
    """
    qa_pairs = []

    for sample in GOLDEN_DATASET:
        context = sample["body_text"]

        # Create question-answer pairs for each policy field
        fields = {
            "policy_insured": "Who is the insured party or policyholder?",
            "line_of_business": "What is the line of business or policy type?",
            "effective_date": "What is the effective date of the policy?",
            "expected_inception_date": "What is the expected inception date for the policy?",
            "target_premium": "What is the target premium amount for the policy?",
        }

        for field, question in fields.items():
            qa_pairs.append(
                {
                    "email_id": sample["email_id"],
                    "question": question,
                    "ground_truth_answer": sample["ground_truth"][field],
                    "context": context,
                }
            )

    return qa_pairs


# Access the QA pairs with this function
def get_evaluation_data():
    """Get the evaluation data for RAGAS."""
    return generate_qa_pairs()
