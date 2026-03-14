"""Test cases for evaluating the RAG system."""

TEST_CASES = [
    {
        "question": "What is the difference between a block and a unit?",
        "expected_sources": ["project_model.md", "project_overview.md"],
        "required_in_answer": ["block", "unit", "sellable"],
        "forbidden_in_answer": [],
        "category": "core",
    },
    {
        "question": "Can invoices exist on blocks?",
        "expected_sources": ["finance_notes.md", "project_overview.md", "project_model.md"],
        "required_in_answer": ["unit"],
        "required_any": False,
        "forbidden_in_answer": ["invoices exist on blocks", "invoices belong to blocks"],
        "category": "contradiction",
    },
    {
        "question": "Who can approve supplier proposals?",
        "expected_sources": ["permission_matrix.txt", "permissions_internal.txt", "supplier_contracts.md", "supplier_workflow"],
        "required_in_answer": ["project manager"],
        "forbidden_in_answer": [],
        "category": "core",
    },
    {
        "question": "What happens when costs are approved after invoicing is completed?",
        "expected_sources": ["invoicing_edge_cases.md", "engeneering_notes.txt", "engeneering_slack_dump.txt"],
        "required_in_answer": ["invoice"],
        "forbidden_in_answer": [],
        "category": "edge_case",
    },
    {
        "question": "What are addendums used for?",
        "expected_sources": ["supplier_contracts.md", "supplier_workflow"],
        "required_in_answer": ["addendum"],
        "forbidden_in_answer": [],
        "category": "core",
    },
    {
        "question": "Which roles can modify financial configuration?",
        "expected_sources": ["permission_matrix.txt", "permissions_internal.txt"],
        "required_in_answer": ["administrator"],
        "forbidden_in_answer": ["project manager can modify financial"],
        "category": "core",
    },
    # Refusal: out-of-scope question (should refuse)
    {
        "question": "How does the platform integrate with Stripe payments?",
        "expected_sources": [],
        "required_in_answer": ["does not contain", "not available", "not mention", "no information", "insufficient information"],
        "required_any": True,
        "forbidden_in_answer": ["stripe integration works", "stripe api"],
        "category": "refusal",
    },
    # Edge case: question that only appears in informal sources
    {
        "question": "What is the INVOICING_COMPLETED flag?",
        "expected_sources": ["invoicing_edge_cases.md"],
        "required_in_answer": ["invoicing"],
        "forbidden_in_answer": [],
        "category": "edge_case",
    },
    # --- Additional test cases for broader coverage ---
    # Contradiction handling: financial admin at unit vs block level
    {
        "question": "At which level does financial administration take place — blocks or units?",
        "expected_sources": ["project_model.md", "project_overview.md", "finance_notes.md"],
        "required_in_answer": ["unit"],
        "required_any": False,
        "forbidden_in_answer": ["financial administration on blocks"],
        "category": "contradiction",
    },
    # Hierarchy knowledge
    {
        "question": "What is the project hierarchy in Easify?",
        "expected_sources": ["project_model.md", "project_overview.md"],
        "required_in_answer": ["project", "block", "unit"],
        "required_any": False,
        "forbidden_in_answer": [],
        "category": "core",
    },
    # Refusal: completely off-topic question
    {
        "question": "What is the weather forecast for tomorrow?",
        "expected_sources": [],
        "required_in_answer": ["does not contain", "not available", "not mention", "no information", "insufficient information"],
        "required_any": True,
        "forbidden_in_answer": [],
        "category": "refusal",
    },
    # Permissions edge case
    {
        "question": "Can a viewer create new projects?",
        "expected_sources": ["permission_matrix.txt", "permissions_internal.txt"],
        "required_in_answer": [],
        "forbidden_in_answer": ["viewers can create"],
        "category": "edge_case",
    },
    # --- 20 new test cases ---
    # 1. Core: invoice lifecycle states
    {
        "question": "What are the possible states of an invoice?",
        "expected_sources": ["finance_notes.md", "finance_rules.txt"],
        "required_in_answer": ["draft", "approved", "sent", "paid"],
        "forbidden_in_answer": [],
        "category": "core",
    },
    # 2. Core: variation costs
    {
        "question": "What are variation costs in Easify?",
        "expected_sources": ["cost_tracking.md", "finance_rules.txt"],
        "required_in_answer": ["additional", "work"],
        "required_any": True,
        "forbidden_in_answer": [],
        "category": "core",
    },
    # 3. Core: margin formula
    {
        "question": "How is project margin calculated?",
        "expected_sources": ["cost_tracking.md"],
        "required_in_answer": ["revenue", "supplier", "cost"],
        "forbidden_in_answer": [],
        "category": "core",
    },
    # 4. Core: what customers can do
    {
        "question": "What actions can a customer perform in Easify?",
        "expected_sources": ["permission_matrix.txt", "permissions_internal.txt"],
        "required_in_answer": ["view", "progress"],
        "forbidden_in_answer": ["approve invoices", "approve supplier"],
        "category": "core",
    },
    # 5. Core: supplier self-approval
    {
        "question": "Can suppliers approve their own proposals?",
        "expected_sources": ["permission_matrix.txt", "permissions_internal.txt", "supplier_contracts.md", "supplier_workflow"],
        "required_in_answer": ["not", "cannot"],
        "required_any": True,
        "forbidden_in_answer": ["suppliers can approve"],
        "category": "core",
    },
    # 6. Core: work items
    {
        "question": "What are work items in the supplier workflow?",
        "expected_sources": ["supplier_contracts.md", "supplier_workflow"],
        "required_in_answer": ["supplier"],
        "forbidden_in_answer": [],
        "category": "core",
    },
    # 7. Core: proposal contents
    {
        "question": "What information does a supplier proposal contain?",
        "expected_sources": ["supplier_contracts.md", "supplier_workflow"],
        "required_in_answer": ["cost", "timeline"],
        "forbidden_in_answer": [],
        "category": "core",
    },
    # 8. Core: margin recalculation triggers
    {
        "question": "When is the project margin recalculated?",
        "expected_sources": ["cost_tracking.md"],
        "required_in_answer": ["proposal", "addendum", "variation"],
        "forbidden_in_answer": [],
        "category": "core",
    },
    # 9. Core: parking spaces as units
    {
        "question": "Are parking spaces considered units in the system?",
        "expected_sources": ["project_overview.md", "project_model.md"],
        "required_in_answer": ["unit"],
        "forbidden_in_answer": [],
        "category": "core",
    },
    # 10. Core: administrator vs project manager
    {
        "question": "What can administrators do that project managers cannot?",
        "expected_sources": ["permission_matrix.txt", "permissions_internal.txt"],
        "required_in_answer": ["system", "role", "configuration"],
        "forbidden_in_answer": [],
        "category": "core",
    },
    # 11. Contradiction: block invoicing from archived docs
    {
        "question": "Can invoices be attached to blocks for easier cost aggregation?",
        "expected_sources": ["archived_specs_2019.md", "historical_changes.md", "finance_notes.md"],
        "required_in_answer": ["unit", "not"],
        "forbidden_in_answer": ["invoices can be attached to blocks", "invoices belong to blocks"],
        "category": "contradiction",
    },
    # 12. Contradiction: block-level reporting vs ownership
    {
        "question": "Do accounting exports group invoices per block?",
        "expected_sources": ["finance_rules.txt", "project_model.md"],
        "required_in_answer": ["unit"],
        "forbidden_in_answer": ["invoices belong to blocks"],
        "category": "contradiction",
    },
    # 13. Edge case: invoices after INVOICING_COMPLETED
    {
        "question": "Can new invoices be created after a unit is marked INVOICING_COMPLETED?",
        "expected_sources": ["invoicing_edge_cases.md", "finance_rules.txt"],
        "required_in_answer": ["yes", "new"],
        "required_any": True,
        "forbidden_in_answer": ["cannot create new invoices", "no new invoices can be created"],
        "category": "edge_case",
    },
    # 14. Edge case: rejected proposals
    {
        "question": "What happens to supplier proposals that are rejected?",
        "expected_sources": ["supplier_contracts.md", "supplier_workflow"],
        "required_in_answer": ["stored", "remain"],
        "required_any": True,
        "forbidden_in_answer": ["deleted", "removed"],
        "category": "edge_case",
    },
    # 15. Edge case: price changes after approval
    {
        "question": "Can supplier prices change after a proposal is approved?",
        "expected_sources": ["supplier_contracts.md", "supplier_workflow"],
        "required_in_answer": ["addendum"],
        "forbidden_in_answer": [],
        "category": "edge_case",
    },
    # 16. Edge case: why block invoicing was abandoned
    {
        "question": "Why was block-level invoicing removed from the system?",
        "expected_sources": ["archived_specs_2019.md", "historical_changes.md", "finance_notes.md"],
        "required_in_answer": ["buyer", "unit", "cost"],
        "required_any": True,
        "forbidden_in_answer": [],
        "category": "edge_case",
    },
    # 17. Edge case: post-completion warning
    {
        "question": "What warning does the system show when costs are approved after invoicing completion?",
        "expected_sources": ["engeneering_notes.txt", "engeneering_slack_dump.txt", "invoicing_edge_cases.md"],
        "required_in_answer": ["warn", "administrator"],
        "required_any": True,
        "forbidden_in_answer": [],
        "category": "edge_case",
    },
    # 18. Core: who marks invoicing complete
    {
        "question": "Who can mark a unit's invoicing as completed?",
        "expected_sources": ["permission_matrix.txt", "permissions_internal.txt"],
        "required_in_answer": ["project manager"],
        "forbidden_in_answer": [],
        "category": "core",
    },
    # 19. Refusal: mobile app
    {
        "question": "Does Easify have a mobile app for iOS and Android?",
        "expected_sources": [],
        "required_in_answer": ["does not contain", "not available", "not mention", "no information", "insufficient information"],
        "required_any": True,
        "forbidden_in_answer": ["easify has a mobile app", "available on ios"],
        "category": "refusal",
    },
    # 20. Refusal: authentication system
    {
        "question": "How does the login and authentication system work in Easify?",
        "expected_sources": [],
        "required_in_answer": ["does not contain", "not available", "not mention", "no information", "insufficient information"],
        "required_any": True,
        "forbidden_in_answer": ["login works by", "authentication uses"],
        "category": "refusal",
    },
]
