"""Seed drug list for the FedSLM drug-repurposing corpus.

`investigated_indications` are *search seeds*, not assertions. They steer the
PubMed / ClinicalTrials.gov queries towards literature where a drug is studied
outside its original approval. Whether an indication is actually approved is
resolved later from the openFDA label text, never from this file.

`hospital` gives the non-IID federated partition (PRD section 5.2):
  A = general medicine / cardiometabolic
  B = pharmacology, oncology, immunology
  C = neurology, psychiatry, paediatrics, surgery
"""

SEED_DRUGS = [
    # ---- Hospital A: general medicine / cardiometabolic ----------------
    {"drug": "metformin", "approved_for": "type 2 diabetes mellitus",
     "investigated_indications": ["cancer", "polycystic ovary syndrome",
                                  "non-alcoholic fatty liver disease", "aging"], "hospital": "A"},
    {"drug": "aspirin", "approved_for": "pain, fever, antiplatelet therapy",
     "investigated_indications": ["colorectal cancer prevention", "preeclampsia"], "hospital": "A"},
    {"drug": "colchicine", "approved_for": "gout",
     "investigated_indications": ["pericarditis", "coronary artery disease"], "hospital": "A"},
    {"drug": "allopurinol", "approved_for": "gout, hyperuricemia",
     "investigated_indications": ["chronic kidney disease", "cardiovascular disease"], "hospital": "A"},
    {"drug": "simvastatin", "approved_for": "hypercholesterolemia",
     "investigated_indications": ["cancer", "sepsis", "multiple sclerosis"], "hospital": "A"},
    {"drug": "atorvastatin", "approved_for": "hypercholesterolemia",
     "investigated_indications": ["cancer", "contrast-induced nephropathy"], "hospital": "A"},
    {"drug": "dapagliflozin", "approved_for": "type 2 diabetes mellitus",
     "investigated_indications": ["heart failure", "chronic kidney disease"], "hospital": "A"},
    {"drug": "empagliflozin", "approved_for": "type 2 diabetes mellitus",
     "investigated_indications": ["heart failure", "chronic kidney disease"], "hospital": "A"},
    {"drug": "liraglutide", "approved_for": "type 2 diabetes mellitus",
     "investigated_indications": ["obesity", "non-alcoholic steatohepatitis"], "hospital": "A"},
    {"drug": "semaglutide", "approved_for": "type 2 diabetes mellitus",
     "investigated_indications": ["obesity", "alcohol use disorder", "Alzheimer disease"], "hospital": "A"},
    {"drug": "spironolactone", "approved_for": "heart failure, hyperaldosteronism",
     "investigated_indications": ["acne vulgaris", "hirsutism"], "hospital": "A"},
    {"drug": "losartan", "approved_for": "hypertension",
     "investigated_indications": ["Marfan syndrome", "aortic aneurysm"], "hospital": "A"},
    {"drug": "verapamil", "approved_for": "hypertension, angina",
     "investigated_indications": ["type 1 diabetes mellitus", "cluster headache"], "hospital": "A"},
    {"drug": "fenofibrate", "approved_for": "dyslipidemia",
     "investigated_indications": ["diabetic retinopathy"], "hospital": "A"},
    {"drug": "pentoxifylline", "approved_for": "intermittent claudication",
     "investigated_indications": ["alcoholic hepatitis", "diabetic nephropathy"], "hospital": "A"},
    {"drug": "ursodeoxycholic acid", "approved_for": "primary biliary cholangitis, gallstones",
     "investigated_indications": ["COVID-19", "cystic fibrosis liver disease"], "hospital": "A"},
    {"drug": "doxycycline", "approved_for": "bacterial infection",
     "investigated_indications": ["rosacea", "abdominal aortic aneurysm", "cancer"], "hospital": "A"},
    {"drug": "cimetidine", "approved_for": "gastroesophageal reflux, peptic ulcer",
     "investigated_indications": ["colorectal cancer"], "hospital": "A"},
    {"drug": "digoxin", "approved_for": "heart failure, atrial fibrillation",
     "investigated_indications": ["cancer"], "hospital": "A"},
    {"drug": "hydroxychloroquine", "approved_for": "malaria, lupus erythematosus",
     "investigated_indications": ["COVID-19", "antiphospholipid syndrome"], "hospital": "A"},

    # ---- Hospital B: pharmacology / oncology / immunology --------------
    {"drug": "thalidomide", "approved_for": "erythema nodosum leprosum",
     "investigated_indications": ["multiple myeloma", "myelodysplastic syndrome"], "hospital": "B"},
    {"drug": "methotrexate", "approved_for": "neoplastic disease",
     "investigated_indications": ["rheumatoid arthritis", "psoriasis", "ectopic pregnancy"], "hospital": "B"},
    {"drug": "rituximab", "approved_for": "non-Hodgkin lymphoma",
     "investigated_indications": ["rheumatoid arthritis", "multiple sclerosis",
                                  "membranous nephropathy"], "hospital": "B"},
    {"drug": "sirolimus", "approved_for": "renal transplant rejection prophylaxis",
     "investigated_indications": ["lymphangioleiomyomatosis", "tuberous sclerosis", "aging"], "hospital": "B"},
    {"drug": "raloxifene", "approved_for": "osteoporosis",
     "investigated_indications": ["breast cancer prevention"], "hospital": "B"},
    {"drug": "tamoxifen", "approved_for": "breast cancer",
     "investigated_indications": ["bipolar mania", "glioma"], "hospital": "B"},
    {"drug": "celecoxib", "approved_for": "osteoarthritis, rheumatoid arthritis",
     "investigated_indications": ["familial adenomatous polyposis", "cancer"], "hospital": "B"},
    {"drug": "disulfiram", "approved_for": "alcohol use disorder",
     "investigated_indications": ["cancer", "glioblastoma"], "hospital": "B"},
    {"drug": "mebendazole", "approved_for": "helminth infection",
     "investigated_indications": ["glioma", "colorectal cancer"], "hospital": "B"},
    {"drug": "niclosamide", "approved_for": "tapeworm infection",
     "investigated_indications": ["cancer", "COVID-19"], "hospital": "B"},
    {"drug": "itraconazole", "approved_for": "fungal infection",
     "investigated_indications": ["basal cell carcinoma", "prostate cancer"], "hospital": "B"},
    {"drug": "auranofin", "approved_for": "rheumatoid arthritis",
     "investigated_indications": ["cancer", "parasitic infection"], "hospital": "B"},
    {"drug": "artesunate", "approved_for": "severe malaria",
     "investigated_indications": ["cancer", "cervical intraepithelial neoplasia"], "hospital": "B"},
    {"drug": "chloroquine", "approved_for": "malaria",
     "investigated_indications": ["cancer", "autophagy inhibition"], "hospital": "B"},
    {"drug": "leflunomide", "approved_for": "rheumatoid arthritis",
     "investigated_indications": ["BK virus nephropathy", "neuroblastoma"], "hospital": "B"},
    {"drug": "zoledronic acid", "approved_for": "osteoporosis, skeletal-related events",
     "investigated_indications": ["breast cancer", "prostate cancer"], "hospital": "B"},
    {"drug": "baricitinib", "approved_for": "rheumatoid arthritis",
     "investigated_indications": ["COVID-19", "alopecia areata", "atopic dermatitis"], "hospital": "B"},
    {"drug": "tocilizumab", "approved_for": "rheumatoid arthritis",
     "investigated_indications": ["COVID-19", "cytokine release syndrome",
                                  "giant cell arteritis"], "hospital": "B"},
    {"drug": "anakinra", "approved_for": "rheumatoid arthritis",
     "investigated_indications": ["COVID-19", "Still disease", "pericarditis"], "hospital": "B"},
    {"drug": "dexamethasone", "approved_for": "inflammatory and allergic conditions",
     "investigated_indications": ["COVID-19", "acute respiratory distress syndrome"], "hospital": "B"},
    {"drug": "ivermectin", "approved_for": "onchocerciasis, strongyloidiasis",
     "investigated_indications": ["COVID-19", "rosacea", "scabies"], "hospital": "B"},
    {"drug": "bevacizumab", "approved_for": "metastatic colorectal cancer",
     "investigated_indications": ["age-related macular degeneration",
                                  "diabetic macular edema"], "hospital": "B"},
    {"drug": "infliximab", "approved_for": "Crohn disease, rheumatoid arthritis",
     "investigated_indications": ["sarcoidosis", "uveitis"], "hospital": "B"},
    {"drug": "zidovudine", "approved_for": "HIV infection",
     "investigated_indications": ["cancer", "adult T-cell leukemia"], "hospital": "B"},

    # ---- Hospital C: neurology / psychiatry / paediatrics / surgery ----
    {"drug": "sildenafil", "approved_for": "erectile dysfunction",
     "investigated_indications": ["pulmonary arterial hypertension", "Raynaud phenomenon",
                                  "Alzheimer disease"], "hospital": "C"},
    {"drug": "minoxidil", "approved_for": "hypertension",
     "investigated_indications": ["androgenetic alopecia"], "hospital": "C"},
    {"drug": "propranolol", "approved_for": "hypertension, angina",
     "investigated_indications": ["infantile hemangioma", "post-traumatic stress disorder",
                                  "migraine"], "hospital": "C"},
    {"drug": "bupropion", "approved_for": "major depressive disorder",
     "investigated_indications": ["smoking cessation", "obesity"], "hospital": "C"},
    {"drug": "ketamine", "approved_for": "anesthesia",
     "investigated_indications": ["treatment-resistant depression", "chronic pain"], "hospital": "C"},
    {"drug": "naltrexone", "approved_for": "opioid and alcohol dependence",
     "investigated_indications": ["fibromyalgia", "Crohn disease", "multiple sclerosis"], "hospital": "C"},
    {"drug": "amantadine", "approved_for": "influenza A prophylaxis",
     "investigated_indications": ["Parkinson disease", "traumatic brain injury"], "hospital": "C"},
    {"drug": "lithium", "approved_for": "bipolar disorder",
     "investigated_indications": ["amyotrophic lateral sclerosis", "Alzheimer disease"], "hospital": "C"},
    {"drug": "valproic acid", "approved_for": "epilepsy",
     "investigated_indications": ["migraine prophylaxis", "bipolar disorder", "cancer"], "hospital": "C"},
    {"drug": "topiramate", "approved_for": "epilepsy",
     "investigated_indications": ["migraine prophylaxis", "obesity", "alcohol use disorder"], "hospital": "C"},
    {"drug": "gabapentin", "approved_for": "epilepsy, postherpetic neuralgia",
     "investigated_indications": ["neuropathic pain", "restless legs syndrome"], "hospital": "C"},
    {"drug": "duloxetine", "approved_for": "major depressive disorder",
     "investigated_indications": ["fibromyalgia", "diabetic peripheral neuropathy"], "hospital": "C"},
    {"drug": "amitriptyline", "approved_for": "depression",
     "investigated_indications": ["migraine prophylaxis", "irritable bowel syndrome",
                                  "neuropathic pain"], "hospital": "C"},
    {"drug": "fluoxetine", "approved_for": "major depressive disorder",
     "investigated_indications": ["cancer", "stroke recovery"], "hospital": "C"},
    {"drug": "prazosin", "approved_for": "hypertension",
     "investigated_indications": ["post-traumatic stress disorder nightmares"], "hospital": "C"},
    {"drug": "memantine", "approved_for": "Alzheimer disease",
     "investigated_indications": ["obsessive-compulsive disorder", "autism"], "hospital": "C"},
    {"drug": "bumetanide", "approved_for": "edema",
     "investigated_indications": ["autism spectrum disorder", "neonatal seizures"], "hospital": "C"},
    {"drug": "finasteride", "approved_for": "benign prostatic hyperplasia",
     "investigated_indications": ["androgenetic alopecia"], "hospital": "C"},
    {"drug": "dimethyl fumarate", "approved_for": "psoriasis",
     "investigated_indications": ["multiple sclerosis", "neuroprotection"], "hospital": "C"},
    {"drug": "botulinum toxin", "approved_for": "cervical dystonia, blepharospasm",
     "investigated_indications": ["chronic migraine", "hyperhidrosis", "overactive bladder"],
     "hospital": "C"},
]

HOSPITAL_LABELS = {
    "A": "general medicine / cardiometabolic",
    "B": "pharmacology / oncology / immunology",
    "C": "neurology / psychiatry / paediatrics",
}


def drug_names():
    return [d["drug"] for d in SEED_DRUGS]


def by_hospital():
    out = {"A": [], "B": [], "C": []}
    for d in SEED_DRUGS:
        out[d["hospital"]].append(d)
    return out


if __name__ == "__main__":
    counts = {k: len(v) for k, v in by_hospital().items()}
    pairs = sum(len(d["investigated_indications"]) for d in SEED_DRUGS)
    print(f"{len(SEED_DRUGS)} seed drugs, {pairs} drug-indication search seeds")
    for k, v in counts.items():
        print(f"  hospital {k} ({HOSPITAL_LABELS[k]}): {v} drugs")
