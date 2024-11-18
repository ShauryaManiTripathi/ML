useless_words = set([
        # Prepositions
        "about", "above", "across", "after", "against", "along", "among", "around", "as", "at",
        "before", "behind", "below", "beneath", "beside", "besides", "between", "beyond", "by",
        "concerning", "despite", "down", "during", "except", "for", "from", "in", "inside", "into",
        "like", "near", "of", "off", "on", "onto", "out", "outside", "over", "past", "regarding",
        "since", "through", "throughout", "to", "toward", "under", "underneath", "until", "unto",
        "up", "upon", "with", "within", "without",
        
        # Filler/Stop words
        "a", "an", "the", "and", "or", "but", "if", "because", "while", "when", "where", "how",
        "so", "then", "than", "that", "which", "who", "whom", "whose", "what", "why", "does",
        "do", "doing", "did", "is", "was", "were", "am", "are", "be", "been", "being", "have",
        "has", "had", "having", "will", "shall", "should", "can", "could", "would", "may", 
        "might", "must",
        
        # Common resume/useless words
        "professional", "work", "responsibility", "project", "skills",
        "detail", "ability", "capability", "dedicated", "self-motivated", 
        "proven", "successfully", "goal-oriented", "efficient", "highly", "excellent", "strong",
        "ability", "communication", "interpersonal", "collaborative", "organized", 
        "detail-oriented", "dynamic",
        
        # Single-letter words
        *list("abcdefghijklmnopqrstuvwxyz")
    ])


professional_terms = {
    # Action/Leadership Verbs
    'achieved', 'administered', 'advised', 'analyzed', 'automated', 'championed',
    'collaborated', 'coordinated', 'decreased', 'delivered', 'demonstrated',
    'developed', 'directed', 'drove', 'enhanced', 'established', 'executed',
    'generated', 'headed', 'implemented', 'improved', 'increased', 'initiated',
    'innovated', 'led', 'managed', 'mentored', 'orchestrated', 'optimized',
    'pioneered', 'reduced', 'restructured', 'streamlined', 'supervised',
    'transformed',

    # Technical/Professional Skills
    'analyzed', 'architected', 'automated', 'designed', 'developed', 'engineered',
    'frameworks', 'implemented', 'integrated', 'methodology', 'optimized',
    'programmed', 'solutions', 'standards', 'systems', 'technical',
    'technologies', 'validated',

    # Project Management
    'agile', 'budget', 'deadlines', 'deliverables', 'kanban', 'milestones',
    'project', 'requirements', 'resources', 'roadmap', 'schedule', 'scrum',
    'sprints', 'stakeholders', 'timeline',

    # Business Impact
    'benchmarks', 'business', 'client', 'conversion', 'cost', 'efficiency',
    'growth', 'impact', 'kpis', 'metrics', 'optimization', 'performance',
    'productivity', 'profit', 'revenue', 'roi', 'sales', 'savings',
    'scalability', 'strategy',

    # Soft Skills/Attributes
    'adaptable', 'analytical', 'collaborative', 'communication', 'creative',
    'detail-oriented', 'dynamic', 'innovative', 'leadership', 'problem-solving',
    'professional', 'strategic', 'team-player', 'versatile',

    # Career Progress
    'advanced', 'awarded', 'certified', 'promoted', 'recognized', 'selected',
    'specialized', 'graduated', 'trained',

    # Achievement Indicators
    'accomplished', 'achieved', 'awarded', 'exceeded', 'outperformed',
    'successful', 'surpassed', 'won',

    # Experience Level
    'expert', 'expertise', 'advanced', 'proficient', 'experienced', 'skilled',
    'specialist', 'professional', 'certified',

    # Team/Interpersonal
    'collaborated', 'committee', 'cross-functional', 'liaison', 'partnership',
    'team', 'teamwork', 'trained', 'mentored', 'supervised'
    }