BROAD_TOPICS = [
    ("broad_topic_001", "Computer Science", ["computing", "software", "algorithm", "data mining", "machine learning", "artificial intelligence"]),
    ("broad_topic_002", "Information Security and Cryptography", ["cryptography", "privacy", "security", "malware", "authentication", "outsourcing"]),
    ("broad_topic_003", "Electronic Engineering", ["vlsi", "circuit", "eda", "integrated circuit", "semiconductor", "chip"]),
    ("broad_topic_004", "Communications and Networking", ["wireless", "network", "mobile", "mimo", "5g", "ad hoc"]),
    ("broad_topic_005", "Electrical Engineering and Energy", ["power", "energy", "grid", "battery", "electric"]),
    ("broad_topic_006", "Mechanical Engineering", ["mechanical", "manufacturing", "robotics", "control"]),
    ("broad_topic_007", "Civil Engineering and Architecture", ["civil", "construction", "architecture", "structural"]),
    ("broad_topic_008", "Environmental Science and Engineering", ["environment", "pollution", "ecotoxicology", "water", "wastewater", "bioremediation"]),
    ("broad_topic_009", "Earth Science and Climate", ["climate", "remote sensing", "geography", "geology", "carbon cycle", "meteorology"]),
    ("broad_topic_010", "Chemistry and Chemical Engineering", ["chemistry", "chemical", "catalysis", "polymer", "materials chemistry"]),
    ("broad_topic_011", "Materials Science", ["materials", "nanomaterials", "nanoparticles", "metallurgy", "composite"]),
    ("broad_topic_012", "Food Science and Technology", ["food", "nutrition", "starch", "protein", "cereal", "fermentation", "food processing"]),
    ("broad_topic_013", "Biology and Life Sciences", ["biology", "microbiology", "genetics", "cell", "molecular", "plant"]),
    ("broad_topic_014", "Agriculture and Animal Science", ["agriculture", "crop", "soil", "animal", "aquaculture"]),
    ("broad_topic_015", "Medicine and Clinical Science", ["medicine", "clinical", "disease", "therapy", "surgery", "patient"]),
    ("broad_topic_016", "Public Health and Epidemiology", ["epidemiology", "public health", "cohort", "mortality", "risk factor", "population"]),
    ("broad_topic_017", "Pharmacy and Pharmacology", ["pharmacy", "pharmacology", "drug", "toxicology", "pharmaceutical"]),
    ("broad_topic_018", "Traditional Chinese Medicine", ["traditional chinese medicine", "acupuncture", "decoction", "herbal", "meridian"]),
    ("broad_topic_019", "Neuroscience and Psychology", ["neuroscience", "psychology", "cognitive", "depression", "brain"]),
    ("broad_topic_020", "Mathematics and Statistics", ["mathematics", "statistics", "probability", "optimization", "modeling"]),
    ("broad_topic_021", "Physics and Astronomy", ["physics", "astronomy", "optics", "quantum"]),
    ("broad_topic_022", "Economics and Management", ["economics", "management", "finance", "innovation", "policy"]),
    ("broad_topic_023", "Social Sciences", ["sociology", "education", "politics", "communication", "social network"]),
    ("broad_topic_024", "Humanities and Arts", ["history", "literature", "language", "philosophy", "art", "opera"]),
]


def broad_topic_candidates():
    return [
        {"id": topic_id, "canonical": canonical, "variants": [canonical] + variants}
        for topic_id, canonical, variants in BROAD_TOPICS
    ]
