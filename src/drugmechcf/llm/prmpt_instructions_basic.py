"""
Prompt templates for DrugMechDB,
used in retrieving Known MoA's,
with targeted disease modes: anonymous (descr only), named only, name + descr
"""

# -----------------------------------------------------------------------------
#   Globals
# -----------------------------------------------------------------------------

DRUGMECHDB_PROMPT_VERSIONS = [
    # --- Version 1 ---
    {
        "version": 1,
        "prompt_persona": "",
        "prompt_goal": ("Given the name or description of a disease, and the name or description of a drug, "
                        "determine whether or not the drug might be useful in treating the disease, "
                        "and if it is useful, provide a Mechanism of Action for the drug."),
        "prompt_instructions": """
1. Say "YES." or "NO supporting Mechanism of Action found." depending on whether the drug might be useful in treating the disease.

2. If the response in step 1 is "YES", provide a Mechanism of Action describing how the drug will affect the disease. Present the Mechanism of Action as a series of interaction steps, one step per line. Each interaction step should mention a direct interaction relationship between two entities, containing the following information:
- source_entity_type: The type of the source entity. One of the following types: [{all_entity_types}]
- source_entity_name: The name of the source entity
- target_entity_type: The type of the target entity. One of the following types: [{all_entity_types}]
- target_entity_name: The name of the target entity
- interaction_relationship: How the source entity interacts with the target entity

Please use as many interaction steps as are needed for a complete response.
Please use the same formal entity name each time the entity is mentioned in your response.

Format each interaction step as:
<source_entity_type>: <source_entity_name> | <interaction_relationship> | <target_entity_type>: <target_entity_name>
""",
        "prompt_examples": [
            {
                "moa_id": "DB00945_MESH_D010146_1",
                "drug_node": "MESH:D001241",
                "drug_name": "Aspirin",
                "disease_node": "MESH:D010146",
                "disease_name": "Pain",
                "disease_association_relation": "causes",
                "example_llm_response": """
YES.

Drug: Aspirin | Decreases Activity Of | Protein: Prostaglandin G/H synthase 1
Drug: Aspirin | Decreases Activity Of | Protein: Prostaglandin G/H synthase 2
Protein: Prostaglandin G/H synthase 1 | Positively Regulates | Biological Process: Prostaglandin Biosynthetic Process
Protein: Prostaglandin G/H synthase 2 | Positively Regulates | Biological Process: Prostaglandin Biosynthetic Process
Biological Process: Prostaglandin Biosynthetic Process | Increases Abundance Of | Chemical: Prostaglandins
Chemical: Prostaglandins | Contributes To | Biological Process: Inflammatory Response
Biological Process: Inflammatory Response | {disease_association_reln} | Disease: {example_disease_name}
"""
            },
            {
                "moa_id": "DB11693_MESH_D014605_1",
                "drug_node": "MESH:C484071",
                "drug_name": "Voclosporin",
                "disease_node": "MESH:D014605",
                "disease_name": "Noninfectious Uveitis",
                "disease_association_relation": "causes",
                "example_llm_response": """
YES.

Drug: Voclosporin | Decreases Activity Of | Protein: Calcineurin subunit B type 1
Protein: Calcineurin subunit B type 1 | Increases Abundance Of | Protein: Nuclear factor of activated T-cells, cytoplasmic 1
Protein: Nuclear factor of activated T-cells, cytoplasmic 1 | Increases Abundance Of | Protein: interleukin 2
Protein: interleukin 2 | Produces | Cell: T-cell
Cell: T-cell | {disease_association_reln} | Disease: {example_disease_name}
"""
            },
            # Example 3: -ive example
            #   DrugMechDB has MoA's for 42 drugs for treating "MESH:D013290"
            #   DrugMechDB has 1 MoA for Drug: Pinacidil (MESH:D020110), for treating Hypertension (MESH:D006973)
            {
                "drug_node": "MESH:D020110",
                "drug_name": "Pinacidil",
                "disease_node": "MESH:D013290",
                "disease_name": "Streptococcus Pyogenes Infection",
                "disease_association_relation": "NOT NEEDED",
                # Special attributes for -ive examples
                "is_negative": True,
                "all_disease_associations": "the Organism: Streptococcus Pyogenes.",
                "filtered_disease_associations": "the Organism: Streptococcus Pyogenes.",
                # --- end of special attributes ---
                "example_llm_response": """
NO supporting Mechanism of Action found.
"""
            }
        ]
     },

    # --- Version 2 ---
    {
        "version": 2,
        "prompt_persona": ("You are a biomedical research assistant, "
                           "assisting in the evaluation of drugs for treating diseases."),
        "prompt_goal": ("Given the name or description of a disease, and the name or description of a drug, "
                        "determine whether or not the drug might be useful in treating the disease, "
                        "and if it is useful, provide a Mechanism of Action for the drug."),
        "prompt_instructions": """
1. Say "YES." or "NO supporting Mechanism of Action found." depending on whether the drug might be useful in treating the disease.

2. If the response in step 1 is "YES", provide a Mechanism of Action describing how the drug will affect the disease.
Present the Mechanism of Action as a series of interaction steps, one step per line.
Each interaction step should mention a direct interaction relationship between two entities, containing 
the following information:
- source_entity_type: The type of the source entity. One of the following types: [{all_entity_types}]
- source_entity_name: The name of the source entity
- target_entity_type: The type of the target entity. One of the following types: [{all_entity_types}]
- target_entity_name: The name of the target entity
- interaction_relationship: How the source entity interacts with the target entity

Please use as many interaction steps as are needed for a complete detailed response, 
using specific entities instead of abstract entities wherever possible.
Please use the same formal entity name each time the entity is mentioned in your response.

Format each interaction step as a triple, using the format:
<source_entity_type>: <source_entity_name> | <interaction_relationship> | <target_entity_type>: <target_entity_name>
""",
        "prompt_examples": [
            # Example 1: +ive example
            {
                "moa_id": "DB00945_MESH_D010146_1",
                "drug_node": "MESH:D001241",
                "drug_name": "Aspirin",
                "disease_node": "MESH:D010146",
                "disease_name": "Pain",
                "disease_association_relation": "causes",
                "example_llm_response": """
YES.

Drug: Aspirin | Decreases Activity Of | Protein: Prostaglandin G/H synthase 1
Drug: Aspirin | Decreases Activity Of | Protein: Prostaglandin G/H synthase 2
Protein: Prostaglandin G/H synthase 1 | Positively Regulates | Biological Process: Prostaglandin Biosynthetic Process
Protein: Prostaglandin G/H synthase 2 | Positively Regulates | Biological Process: Prostaglandin Biosynthetic Process
Biological Process: Prostaglandin Biosynthetic Process | Increases Abundance Of | Chemical: Prostaglandins
Chemical: Prostaglandins | Contributes To | Biological Process: Inflammatory Response
Biological Process: Inflammatory Response | {disease_association_reln} | Disease: {example_disease_name}
"""
            },
            # Example 2: +ive example
            {
                "moa_id": "DB11693_MESH_D014605_1",
                "drug_node": "MESH:C484071",
                "drug_name": "Voclosporin",
                "disease_node": "MESH:D014605",
                "disease_name": "Noninfectious Uveitis",
                "disease_association_relation": "causes",
                "example_llm_response": """
YES.

Drug: Voclosporin | Decreases Activity Of | Protein: Calcineurin subunit B type 1
Protein: Calcineurin subunit B type 1 | Increases Abundance Of | Protein: Nuclear factor of activated T-cells, cytoplasmic 1
Protein: Nuclear factor of activated T-cells, cytoplasmic 1 | Increases Abundance Of | Protein: interleukin 2
Protein: interleukin 2 | Produces | Cell: T-cell
Cell: T-cell | {disease_association_reln} | Disease: {example_disease_name}
"""
            },
            # Example 3: -ive example
            #   DrugMechDB has MoA's for 42 drugs for treating "MESH:D013290"
            #   DrugMechDB has 1 MoA for Drug: Pinacidil (MESH:D020110), for treating Hypertension (MESH:D006973)
            {
                "drug_node": "MESH:D020110",
                "drug_name": "Pinacidil",
                "disease_node": "MESH:D013290",
                "disease_name": "Streptococcus Pyogenes Infection",
                # Special attributes for -ive examples
                "is_negative": True,
                # ... for use with PromptStyle.ANONYMIZED_DISEASE
                "all_disease_associations": "the Organism: Streptococcus Pyogenes.",
                # ... for use with PromptStyle.NAMED_DISEASE_WITH_ALL_ASSOCIATIONS
                "filtered_disease_associations": "the Organism: Streptococcus Pyogenes.",
                # --- end of special attributes ---
                "example_llm_response": """
NO supporting Mechanism of Action found.
"""
            },
            # Example 4:
            # min-length = 6
            # Entity types in graph = BiologicalProcess, ChemicalSubstance, Disease, Drug, GeneFamily,
            #                         Pathway, PhenotypicFeature, Protein
            {
                "moa_id": "DB00195_MESH_D006973_1",
                "drug_node": "MESH:D015784",
                "drug_name": "Betaxolol",
                "disease_node": "MESH:D006973",
                "disease_name": "Hypertensive Disorder",
                "disease_association_relation": "manifestation of",
                "example_llm_response": """
YES.

Drug: Betaxolol | Decreases Activity Of | Protein: Beta-1 adrenergic receptor
Protein: Beta-1 adrenergic receptor | Participates In | Pathway: G Alpha (s) Signalling Events
Pathway: G Alpha (s) Signalling Events | Increases Abundance Of | Chemical: 3,5-cyclic AMP
Pathway: G Alpha (s) Signalling Events | Positively Regulates | Gene: Voltage-dependent calcium channel, L-type, alpha-1 subunit
Chemical: 3,5-cyclic AMP | Positively Correlated With | Biological Process: Heart Contraction
Gene: Voltage-dependent calcium channel, L-type, alpha-1 subunit | Positively Correlated With | Biological Process: Heart Contraction
Biological Process: Heart Contraction | Positively Correlated With | Phenotype: Increased Blood Pressure
Phenotype: Increased Blood Pressure | {disease_association_reln} | Disease: {example_disease_name}
"""
            },
            # Example 5:
            # min-length = 4
            # Entity types in graph = BiologicalProcess, Disease, Drug, GeneFamily, Pathway, Protein
            {
                "moa_id": "DB09079_MESH_D000080203_1",
                "drug_node": "MESH:C530716",
                "drug_name": "Nintedanib",
                "disease_node": "MESH:D000080203",
                "disease_name": "Hamman-Rich Syndrome",
                "disease_association_relation": "contributes to",
                "example_llm_response": """
YES.

Drug: Nintedanib | Decreases Activity Of | Protein: Tyrosine-protein phosphatase non-receptor type
Drug: Nintedanib | Decreases Activity Of | Protein: Vascular endothelial growth factor receptor 1
Drug: Nintedanib | Decreases Activity Of | Protein: Vascular endothelial growth factor receptor 2
Drug: Nintedanib | Decreases Activity Of | Protein: Vascular endothelial growth factor receptor 3
Drug: Nintedanib | Decreases Activity Of | Gene: Platelet-derived growth factor, N-terminal
Drug: Nintedanib | Decreases Activity Of | Gene: Fibroblast growth factor receptor family
Gene: Platelet-derived growth factor, N-terminal | Participates In | Pathway: Signaling By PDGF
Gene: Fibroblast growth factor receptor family | Participates In | Pathway: Signaling By FGFR
Protein: Vascular endothelial growth factor receptor 1 | Participates In | Pathway: VEGFR2 Mediated Cell Proliferation
Protein: Tyrosine-protein phosphatase non-receptor type | Participates In | Pathway: Signal Transduction
Protein: Vascular endothelial growth factor receptor 3 | Participates In | Pathway: VEGFR2 Mediated Cell Proliferation
Protein: Vascular endothelial growth factor receptor 2 | Participates In | Pathway: VEGFR2 Mediated Cell Proliferation
Pathway: Signal Transduction | Positively Regulates | Biological Process: Fibroblast Proliferation
Pathway: Signaling By PDGF | Positively Regulates | Biological Process: Fibroblast Proliferation
Pathway: Signaling By FGFR | Positively Regulates | Biological Process: Fibroblast Proliferation
Pathway: VEGFR2 Mediated Cell Proliferation | Positively Regulates | Biological Process: Fibroblast Proliferation
Biological Process: Fibroblast Proliferation | {disease_association_reln} | Disease: {example_disease_name}
"""
            }
        ]
    },
]
