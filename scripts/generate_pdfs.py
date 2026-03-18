"""Generate 9 synthetic plant floor PDFs (3 per domain) using reportlab."""
import os
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

styles = getSampleStyleSheet()
title_style = ParagraphStyle("Title2", parent=styles["Title"], fontSize=18, spaceAfter=20)
h1_style = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=14, spaceAfter=10)
body_style = ParagraphStyle("Body2", parent=styles["Normal"], fontSize=10, spaceAfter=6)


def build_pdf(path: str, title: str, revision: str, sections: list[tuple[str, list[str]]]) -> None:
    doc = SimpleDocTemplate(path, pagesize=LETTER,
                            leftMargin=inch, rightMargin=inch,
                            topMargin=inch, bottomMargin=inch)
    story = []
    story.append(Paragraph(title, title_style))
    story.append(Paragraph(f"Revision: {revision}", body_style))
    story.append(Spacer(1, 0.3 * inch))
    for sec_title, items in sections:
        story.append(Paragraph(sec_title, h1_style))
        for item in items:
            story.append(Paragraph(item, body_style))
        story.append(Spacer(1, 0.2 * inch))
    doc.build(story)


# ── Safety PDFs ──────────────────────────────────────────────────────────────

LOTO_SECTIONS = [
    ("1. Scope", [
        "1.1 This procedure applies to all electrical and hydraulic equipment on the plant floor.",
        "1.2 All maintenance personnel must complete LOTO training before performing any isolation.",
        "1.3 Applicable equipment includes: conveyor drives (480 VAC), hydraulic press units (3000 PSI), CNC spindle motors.",
        "1.4 GHS hazard code: GHS06 (Toxic), GHS08 (Health Hazard) apply to hydraulic fluid handling.",
        "1.5 NFPA 70E arc-flash boundary requirements must be observed for all electrical work.",
    ]),
    ("2. Procedures", [
        "2.1 Notify affected employees before initiating lockout.",
        "2.2 Identify all energy sources: electrical (480 VAC / 120 VAC), hydraulic (3000 PSI), pneumatic (90 PSI).",
        "2.3 Shut down equipment using normal stopping procedure.",
        "2.4 Isolate energy sources: open main disconnect switch MDS-101, close hydraulic isolation valve HIV-22.",
        "2.5 Apply personal lock to each energy-isolating device. Tag with DANGER — DO NOT OPERATE.",
        "2.6 Release stored energy: bleed hydraulic lines to 0 PSI, discharge capacitor bank CAP-03.",
        "2.7 Verify zero energy state with calibrated multimeter (Fluke 87V) and pressure gauge PG-07.",
        "2.8 Perform maintenance task.",
        "2.9 Remove all tools and materials from equipment.",
        "2.10 Remove personal lock and tag; restore energy in reverse order.",
    ]),
    ("3. Requirements", [
        "3.1 PPE required: Class E hard hat, safety glasses (ANSI Z87.1), arc-flash suit (8 cal/cm²) for electrical work.",
        "3.2 Hydraulic work requires nitrile gloves (EN 374) and face shield.",
        "3.3 All locks must be individually keyed; master keys are prohibited.",
        "3.4 LOTO log must be completed (Form SF-001) before and after each isolation.",
        "3.5 Annual LOTO audit required per OSHA 29 CFR 1910.147.",
    ]),
    ("4. References", [
        "4.1 OSHA 29 CFR 1910.147 — Control of Hazardous Energy.",
        "4.2 NFPA 70E — Standard for Electrical Safety in the Workplace.",
        "4.3 GHS Safety Data Sheet — Hydraulic Fluid ISO 46.",
        "4.4 Plant Equipment Register Rev. 12.",
        "4.5 Form SF-001 — LOTO Log Sheet.",
    ]),
]

EMERGENCY_SECTIONS = [
    ("1. Scope", [
        "1.1 This guide covers fire, chemical spill, and personal injury emergencies on the plant floor.",
        "1.2 All employees must review this guide annually and during onboarding.",
        "1.3 Emergency contact: Plant Safety Officer ext. 5100; External: 911.",
        "1.4 NFPA 10 portable fire extinguisher placement: one per 75 ft² in high-hazard areas.",
        "1.5 Chemical inventory includes: hydraulic fluid (GHS02 Flammable), cutting coolant (GHS07 Irritant).",
    ]),
    ("2. Procedures", [
        "2.1 FIRE: Activate nearest pull station. Evacuate via marked exits. Do not use elevators.",
        "2.2 FIRE: If trained, use Class ABC extinguisher (PASS: Pull, Aim, Squeeze, Sweep) on small fires only.",
        "2.3 FIRE: Assembly point: Parking Lot B, north gate. Supervisor takes headcount.",
        "2.4 CHEMICAL SPILL: Don nitrile gloves and safety goggles before approaching spill.",
        "2.5 CHEMICAL SPILL: Contain spill with absorbent pads (spill kit SK-02 at each workstation).",
        "2.6 CHEMICAL SPILL: Dispose of contaminated material in labeled hazardous waste drum HW-04.",
        "2.7 INJURY: Call ext. 5100 immediately. Do not move injured person unless in immediate danger.",
        "2.8 INJURY: Apply first aid (AED located at Station 3, first aid kit at Station 7).",
        "2.9 INJURY: Complete incident report Form IR-002 within 24 hours.",
        "2.10 All emergencies must be logged in the Emergency Response Log (Form ER-001).",
    ]),
    ("3. Requirements", [
        "3.1 Evacuation routes must be kept clear at all times (minimum 28-inch aisle width per NFPA 101).",
        "3.2 Fire extinguishers inspected monthly; annual service by certified technician.",
        "3.3 Spill kits restocked within 24 hours of use.",
        "3.4 Emergency drills conducted quarterly; records retained for 3 years.",
        "3.5 All employees must know location of nearest AED and first aid kit.",
    ]),
    ("4. References", [
        "4.1 NFPA 10 — Standard for Portable Fire Extinguishers.",
        "4.2 NFPA 101 — Life Safety Code.",
        "4.3 OSHA 29 CFR 1910.38 — Emergency Action Plans.",
        "4.4 GHS SDS — Hydraulic Fluid ISO 46; Cutting Coolant MX-200.",
        "4.5 Form IR-002 — Incident Report; Form ER-001 — Emergency Response Log.",
    ]),
]

PPE_SECTIONS = [
    ("1. Scope", [
        "1.1 This standard defines PPE selection, inspection, and disposal for all plant floor hazards.",
        "1.2 Hazard assessment required per OSHA 29 CFR 1910.132 before PPE selection.",
        "1.3 PPE is the last line of defense; engineering and administrative controls take priority.",
        "1.4 All PPE must meet applicable ANSI/ISEA standards.",
        "1.5 PPE inventory managed by Safety Coordinator; reorder point: 20% of par level.",
    ]),
    ("2. Procedures", [
        "2.1 Eye/Face: Safety glasses (ANSI Z87.1) required in all production areas.",
        "2.2 Eye/Face: Face shield required for grinding, chemical handling, and hydraulic work.",
        "2.3 Hand: Nitrile gloves (EN 374) for chemical contact; cut-resistant gloves (ANSI A4) for sharp edges.",
        "2.4 Foot: Steel-toed boots (ASTM F2413) required on all plant floor areas.",
        "2.5 Head: Class E hard hat required near overhead work and electrical panels.",
        "2.6 Hearing: Foam earplugs (NRR 29) required in areas > 85 dBA (marked with yellow floor tape).",
        "2.7 Respiratory: N95 respirator for dust; half-face APF-10 for chemical vapors.",
        "2.8 Inspect PPE before each use: check for cracks, tears, discoloration, or deformation.",
        "2.9 Damaged PPE must be removed from service immediately and tagged OUT OF SERVICE.",
        "2.10 Dispose of single-use PPE in designated waste bins; reusable PPE cleaned per manufacturer spec.",
    ]),
    ("3. Requirements", [
        "3.1 PPE training required at hire and annually thereafter; records in Form PPE-003.",
        "3.2 Hard hats replaced every 5 years or after impact, whichever comes first.",
        "3.3 Safety glasses replaced when lenses are scratched or frames are bent.",
        "3.4 Gloves inspected before each use; replaced at first sign of degradation.",
        "3.5 Hearing protection replaced when foam no longer expands to fill ear canal.",
    ]),
    ("4. References", [
        "4.1 OSHA 29 CFR 1910.132–138 — Personal Protective Equipment.",
        "4.2 ANSI Z87.1 — Eye and Face Protection.",
        "4.3 ANSI/ISEA 105 — Hand Protection.",
        "4.4 ASTM F2413 — Foot Protection.",
        "4.5 Form PPE-003 — PPE Training Record.",
    ]),
]


# ── Maintenance PDFs ──────────────────────────────────────────────────────────

CONVEYOR_SECTIONS = [
    ("1. Scope", [
        "1.1 This manual covers scheduled maintenance for conveyor drive motors (Model: Baldor EM3770T, 10 HP, 460 VAC).",
        "1.2 Applicable conveyor lines: CV-01 through CV-08 in Assembly Bay A and B.",
        "1.3 Maintenance intervals: daily inspection, weekly lubrication, monthly torque check, annual overhaul.",
        "1.4 Spare parts stocked: bearing 6205-2RS (P/N: BRG-6205), V-belt B68 (P/N: VBT-B68), shaft seal (P/N: SS-3770).",
        "1.5 All work must be performed under LOTO per procedure SF-LOTO-001.",
    ]),
    ("2. Procedures", [
        "2.1 DAILY: Inspect motor housing for unusual vibration or noise. Record in Maintenance Log ML-CV.",
        "2.2 DAILY: Check belt tension: deflection must be 1/2 inch per foot of span at 5 lbf.",
        "2.3 WEEKLY: Lubricate drive-end bearing with Mobilux EP2 grease, 2 pumps per fitting.",
        "2.4 WEEKLY: Inspect V-belt for cracking, fraying, or glazing; replace if wear exceeds 1/8 inch.",
        "2.5 MONTHLY: Torque motor mounting bolts to 45 ft-lbs (M12 Grade 8.8) using calibrated torque wrench TW-02.",
        "2.6 MONTHLY: Measure motor winding resistance: phase-to-phase must be within 5% of nameplate value.",
        "2.7 MONTHLY: Check coupling alignment: angular misalignment < 0.5°, parallel < 0.005 inch.",
        "2.8 ANNUAL: Replace bearings 6205-2RS (P/N: BRG-6205) regardless of condition.",
        "2.9 ANNUAL: Megger test motor windings: insulation resistance > 100 MΩ at 500 VDC.",
        "2.10 ANNUAL: Replace shaft seal (P/N: SS-3770) and repack with Mobilux EP2.",
    ]),
    ("3. Requirements", [
        "3.1 Motor operating temperature must not exceed 40°C ambient + 80°C rise (Class F insulation).",
        "3.2 Vibration limit: 0.1 in/s RMS at motor bearing housing (ISO 10816-3 Zone A).",
        "3.3 Belt replacement threshold: elongation > 1% of nominal length or visible cracking.",
        "3.4 All maintenance recorded in CMMS work order; parts usage logged against asset CV-01 to CV-08.",
        "3.5 Torque wrench TW-02 calibrated annually; calibration sticker must be current.",
    ]),
    ("4. References", [
        "4.1 Baldor EM3770T Motor Installation and Maintenance Manual.",
        "4.2 ISO 10816-3 — Mechanical Vibration Evaluation.",
        "4.3 OSHA 29 CFR 1910.147 — LOTO.",
        "4.4 Mobilux EP2 Product Data Sheet.",
        "4.5 CMMS Asset Register — Conveyor Lines CV-01 to CV-08.",
    ]),
]

HYDRAULIC_SECTIONS = [
    ("1. Scope", [
        "1.1 This manual covers inspection and maintenance of hydraulic press units HP-01 and HP-02 (Schuler Model SP-200, 200-ton capacity).",
        "1.2 Operating pressure: 3000 PSI nominal; relief valve set at 3300 PSI.",
        "1.3 Hydraulic fluid: Shell Tellus S2 M 46 (ISO VG 46); reservoir capacity: 50 gallons.",
        "1.4 Maintenance intervals: daily fluid check, monthly filter replacement, annual fluid change.",
        "1.5 All work performed under LOTO per SF-LOTO-001; hydraulic pressure must be bled to 0 PSI before service.",
    ]),
    ("2. Procedures", [
        "2.1 DAILY: Check fluid level in sight glass SG-01; level must be between MIN and MAX marks.",
        "2.2 DAILY: Inspect all hydraulic lines and fittings for leaks; record any leaks in Log ML-HP.",
        "2.3 DAILY: Verify relief valve RV-01 setting with calibrated gauge PG-07: 3300 ± 50 PSI.",
        "2.4 MONTHLY: Replace return-line filter element (P/N: FLT-10-M60X) — 10 micron, 60 GPM rated.",
        "2.5 MONTHLY: Sample fluid for particle count analysis; ISO 4406 cleanliness target: 16/14/11.",
        "2.6 MONTHLY: Inspect cylinder seals for weeping; replace if leak rate > 1 drop per 10 cycles.",
        "2.7 ANNUAL: Drain and flush reservoir; replace all fluid with Shell Tellus S2 M 46.",
        "2.8 ANNUAL: Replace all hose assemblies older than 5 years regardless of condition.",
        "2.9 ANNUAL: Calibrate pressure transducer PT-01 against NIST-traceable reference gauge.",
        "2.10 TROUBLESHOOTING: Slow press cycle — check pump output (should be 15 GPM at 3000 PSI); worn pump if < 12 GPM.",
    ]),
    ("3. Requirements", [
        "3.1 Fluid temperature must not exceed 140°F (60°C) during operation; high-temp alarm at 150°F.",
        "3.2 Pressure tolerance: ±50 PSI of set point at steady state.",
        "3.3 Filter replacement mandatory at 500 operating hours or monthly, whichever comes first.",
        "3.4 Fluid cleanliness ISO 4406 level must not exceed 18/16/13; shut down if exceeded.",
        "3.5 All pressure gauges calibrated annually; calibration records retained 3 years.",
    ]),
    ("4. References", [
        "4.1 Schuler SP-200 Hydraulic Press Service Manual Rev. 4.",
        "4.2 Shell Tellus S2 M 46 Product Data Sheet.",
        "4.3 ISO 4406 — Hydraulic Fluid Cleanliness Classification.",
        "4.4 OSHA 29 CFR 1910.147 — LOTO.",
        "4.5 CMMS Asset Register — HP-01, HP-02.",
    ]),
]

CNC_SECTIONS = [
    ("1. Scope", [
        "1.1 This manual covers calibration and maintenance for CNC machining centers MC-01 through MC-04 (Haas VF-2SS).",
        "1.2 Spindle speed range: 50–12,000 RPM; max spindle power: 30 HP.",
        "1.3 Coolant system: semi-synthetic coolant (Blaser Swisslube Blasocut 4000 Strong), 20-gallon sump.",
        "1.4 Maintenance intervals: daily coolant check, weekly spindle warm-up, monthly axis calibration, annual spindle rebuild.",
        "1.5 All work performed under LOTO per SF-LOTO-001.",
    ]),
    ("2. Procedures", [
        "2.1 DAILY: Check coolant concentration with refractometer; target 8–10% (Brix 4–5).",
        "2.2 DAILY: Inspect coolant for tramp oil; skim if oil layer > 1/8 inch.",
        "2.3 DAILY: Run spindle warm-up program O0001: 500 RPM × 5 min, 3000 RPM × 5 min, 8000 RPM × 5 min.",
        "2.4 WEEKLY: Lubricate linear guideways with Mobil Vactra No. 2 oil via auto-lube system; verify 2 cc per cycle.",
        "2.5 WEEKLY: Check tool changer arm for smooth operation; lubricate pivot pin with Mobilux EP2.",
        "2.6 MONTHLY: Calibrate X, Y, Z axes using Renishaw QC20-W ballbar; circularity tolerance ≤ 0.010 mm.",
        "2.7 MONTHLY: Replace coolant sump filter (P/N: FLT-COOL-20) and top up coolant to MAX mark.",
        "2.8 MONTHLY: Inspect spindle taper (CAT 40) for fretting or corrosion; clean with lint-free cloth and light oil.",
        "2.9 ANNUAL: Rebuild spindle bearings (P/N: BRG-7014-AC); replace if runout > 0.002 mm TIR.",
        "2.10 ANNUAL: Replace coolant sump entirely; clean tank with disinfectant to prevent bacterial growth.",
    ]),
    ("3. Requirements", [
        "3.1 Spindle runout must not exceed 0.002 mm TIR at spindle nose.",
        "3.2 Axis positioning accuracy: ±0.005 mm over full travel (ISO 230-2).",
        "3.3 Coolant pH must be maintained between 8.5 and 9.5; test weekly with pH strips.",
        "3.4 Coolant concentration outside 6–12% range requires immediate correction.",
        "3.5 Ballbar test results archived in CMMS; trend analysis performed quarterly.",
    ]),
    ("4. References", [
        "4.1 Haas VF-2SS Operator and Service Manual.",
        "4.2 Renishaw QC20-W Ballbar System User Guide.",
        "4.3 Blaser Swisslube Blasocut 4000 Strong Product Data Sheet.",
        "4.4 ISO 230-2 — Test Code for Machine Tools.",
        "4.5 CMMS Asset Register — MC-01 to MC-04.",
    ]),
]


# ── Quality PDFs ──────────────────────────────────────────────────────────────

INCOMING_SECTIONS = [
    ("1. Scope", [
        "1.1 This standard defines incoming material inspection procedures for all purchased parts and raw materials.",
        "1.2 Applies to: steel bar stock, machined castings, electronic assemblies, and consumables.",
        "1.3 Sampling plan based on ANSI/ASQ Z1.4 (attribute) and Z1.9 (variable) standards.",
        "1.4 Inspection level: Normal Level II for standard suppliers; Tightened Level II for new or probationary suppliers.",
        "1.5 Non-conforming material tagged with RED hold tag and quarantined in Hold Area QA-H1.",
    ]),
    ("2. Procedures", [
        "2.1 Receive shipment; verify purchase order number, quantity, and part number against packing slip.",
        "2.2 Inspect packaging for damage; photograph any damage before opening.",
        "2.3 Select sample per ANSI/ASQ Z1.4 Table II-A (AQL 1.0 for critical, 2.5 for major, 4.0 for minor).",
        "2.4 Dimensional check: measure OD, ID, length, and flatness per drawing tolerances using calibrated CMM.",
        "2.5 Steel bar stock: verify material cert (MTR) against ASTM A36 or A108 as specified; check heat number.",
        "2.6 Castings: check surface finish Ra ≤ 3.2 µm on mating surfaces; no porosity > 1 mm diameter.",
        "2.7 Electronic assemblies: verify RoHS compliance certificate; inspect solder joints per IPC-A-610 Class 2.",
        "2.8 Record all measurements in Incoming Inspection Report Form QC-IIR-001.",
        "2.9 Accept lot if zero defects found in sample (c=0 plan); reject if one or more defects found.",
        "2.10 Accepted material: apply GREEN acceptance label with date, inspector ID, and lot number.",
    ]),
    ("3. Requirements", [
        "3.1 Dimensional tolerances: ±0.005 inch for machined parts; ±0.010 inch for castings unless otherwise specified.",
        "3.2 Surface finish: Ra ≤ 1.6 µm for bearing surfaces; Ra ≤ 3.2 µm for general surfaces.",
        "3.3 All measuring equipment calibrated per ISO/IEC 17025; calibration sticker must be current.",
        "3.4 Inspection records retained for 7 years or product life, whichever is longer.",
        "3.5 Supplier corrective action (SCAR) issued for any rejected lot; response required within 10 business days.",
    ]),
    ("4. References", [
        "4.1 ANSI/ASQ Z1.4 — Sampling Procedures for Attributes.",
        "4.2 ANSI/ASQ Z1.9 — Sampling Procedures for Variables.",
        "4.3 ASTM A36 / A108 — Carbon Steel Standards.",
        "4.4 IPC-A-610 — Acceptability of Electronic Assemblies.",
        "4.5 Form QC-IIR-001 — Incoming Inspection Report.",
    ]),
]

DEFECT_SECTIONS = [
    ("1. Scope", [
        "1.1 This guide defines defect codes, severity classifications, and disposition procedures for all production defects.",
        "1.2 Applies to all in-process and final inspection activities.",
        "1.3 Defect severity levels: Critical (Class 1), Major (Class 2), Minor (Class 3).",
        "1.4 Critical defects: safety hazard or non-functional product — zero tolerance, 100% inspection required.",
        "1.5 All defects recorded in Quality Management System (QMS) with defect code, quantity, and disposition.",
    ]),
    ("2. Procedures", [
        "2.1 CRITICAL (Class 1) defects: immediately quarantine affected lot; notify Quality Manager and Production Supervisor.",
        "2.2 Class 1 examples: DC-001 Dimensional out-of-tolerance on safety-critical feature; DC-002 Missing safety label.",
        "2.3 Class 1 disposition: 100% sort, rework if possible, or scrap; no ship authorization without QM sign-off.",
        "2.4 MAJOR (Class 2) defects: segregate affected parts; initiate NCR (Non-Conformance Report Form QC-NCR-002).",
        "2.5 Class 2 examples: DC-010 Surface scratch > 0.5 mm depth on functional surface; DC-011 Thread damage.",
        "2.6 Class 2 disposition: rework, use-as-is with engineering deviation, or scrap; MRB review within 48 hours.",
        "2.7 MINOR (Class 3) defects: document in QMS; no immediate production stop required.",
        "2.8 Class 3 examples: DC-020 Cosmetic scratch on non-functional surface; DC-021 Minor burr on non-mating edge.",
        "2.9 Class 3 disposition: rework at next available opportunity or accept with customer concession.",
        "2.10 All dispositions require inspector signature, date, and QMS entry within 24 hours of detection.",
    ]),
    ("3. Requirements", [
        "3.1 Defect codes DC-001 through DC-099 reserved for Critical; DC-100 through DC-199 for Major; DC-200+ for Minor.",
        "3.2 NCR must be opened for all Class 1 and Class 2 defects within 4 hours of detection.",
        "3.3 Root cause analysis (5-Why or Fishbone) required for all Class 1 defects within 5 business days.",
        "3.4 Corrective action effectiveness verified at 30, 60, and 90 days.",
        "3.5 Defect trend reports generated monthly; Pareto analysis identifies top 5 defect codes.",
    ]),
    ("4. References", [
        "4.1 ISO 9001:2015 — Quality Management Systems.",
        "4.2 AIAG PPAP — Production Part Approval Process.",
        "4.3 Form QC-NCR-002 — Non-Conformance Report.",
        "4.4 Form QC-RCA-003 — Root Cause Analysis.",
        "4.5 QMS Defect Code Master List Rev. 8.",
    ]),
]

FINAL_INSPECTION_SECTIONS = [
    ("1. Scope", [
        "1.1 This procedure defines final product inspection criteria before shipment authorization.",
        "1.2 Applies to all finished goods produced at this facility.",
        "1.3 Sampling plan: ANSI/ASQ Z1.4 Normal Level II, AQL 1.0 for critical, 2.5 for major dimensions.",
        "1.4 100% inspection required for: safety-critical features, customer-specified characteristics, and first articles.",
        "1.5 Final inspection authority rests with Quality Inspector Level II or higher.",
    ]),
    ("2. Procedures", [
        "2.1 Retrieve approved drawing and inspection plan from QMS for the part number being inspected.",
        "2.2 Verify part number, revision level, and quantity against traveler and work order.",
        "2.3 Dimensional inspection: measure all critical dimensions (marked with balloon numbers on drawing) using CMM.",
        "2.4 CMM program: load part-specific program from server \\\\QA-SERVER\\CMM-Programs\\; fixture per setup sheet.",
        "2.5 Acceptance criteria: all critical dimensions within ±0.005 inch; major dimensions within ±0.010 inch.",
        "2.6 Visual inspection: check surface finish, marking, labeling, and packaging per customer spec.",
        "2.7 Functional test (if applicable): verify assembly fits, torque values, and electrical continuity per test spec.",
        "2.8 Record all measurements in Final Inspection Report Form QC-FIR-004.",
        "2.9 ACCEPT: apply GREEN ship label; update QMS lot status to RELEASED; notify shipping.",
        "2.10 REJECT: apply RED hold tag; open NCR Form QC-NCR-002; notify Production Supervisor.",
    ]),
    ("3. Requirements", [
        "3.1 All critical dimensions must be within tolerance; no exceptions without approved engineering deviation.",
        "3.2 Surface finish Ra ≤ 1.6 µm on bearing and sealing surfaces; verified with profilometer.",
        "3.3 Marking and labeling must match customer purchase order requirements exactly.",
        "3.4 CMM calibration verified daily with reference sphere; full calibration annually per ISO/IEC 17025.",
        "3.5 Final inspection records retained for 10 years or product life, whichever is longer.",
    ]),
    ("4. References", [
        "4.1 ANSI/ASQ Z1.4 — Sampling Procedures for Attributes.",
        "4.2 ISO/IEC 17025 — Calibration Laboratory Requirements.",
        "4.3 Form QC-FIR-004 — Final Inspection Report.",
        "4.4 Form QC-NCR-002 — Non-Conformance Report.",
        "4.5 Customer-Specific Requirements Register Rev. 5.",
    ]),
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    base = os.path.join(os.path.dirname(__file__), "..", "data", "pdfs")
    dirs = {
        "safety_procedures": base + "/safety_procedures",
        "maintenance_manuals": base + "/maintenance_manuals",
        "quality_control_standards": base + "/quality_control_standards",
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    pdfs = [
        (dirs["safety_procedures"] + "/lockout_tagout_procedure.pdf",
         "Lockout/Tagout Procedure", "Rev. 3", LOTO_SECTIONS),
        (dirs["safety_procedures"] + "/emergency_response_guide.pdf",
         "Emergency Response Guide", "Rev. 2", EMERGENCY_SECTIONS),
        (dirs["safety_procedures"] + "/ppe_standards.pdf",
         "PPE Selection and Standards", "Rev. 4", PPE_SECTIONS),
        (dirs["maintenance_manuals"] + "/conveyor_motor_maintenance.pdf",
         "Conveyor Motor Maintenance Manual", "Rev. 5", CONVEYOR_SECTIONS),
        (dirs["maintenance_manuals"] + "/hydraulic_press_manual.pdf",
         "Hydraulic Press Service Manual", "Rev. 4", HYDRAULIC_SECTIONS),
        (dirs["maintenance_manuals"] + "/cnc_machine_maintenance.pdf",
         "CNC Machine Maintenance Manual", "Rev. 3", CNC_SECTIONS),
        (dirs["quality_control_standards"] + "/incoming_inspection_standards.pdf",
         "Incoming Inspection Standards", "Rev. 6", INCOMING_SECTIONS),
        (dirs["quality_control_standards"] + "/defect_classification_guide.pdf",
         "Defect Classification Guide", "Rev. 8", DEFECT_SECTIONS),
        (dirs["quality_control_standards"] + "/final_inspection_procedure.pdf",
         "Final Inspection Procedure", "Rev. 5", FINAL_INSPECTION_SECTIONS),
    ]

    for path, title, revision, sections in pdfs:
        build_pdf(path, title, revision, sections)
        print(f"Generated: {path}")

    print("Done. 9 PDFs generated.")


if __name__ == "__main__":
    main()
