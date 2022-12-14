domains = [{'description': '(opt18) Settlers:\nsubmitted by Marcel Steinmetz\nSettlers was a numeric domain originally developed by Patrik Haslum and included in IPC\n2002. This is a variant where numeric resources are discretized and numeric effects are\nencoded using quantified conditional effects. Moreover, this is a resource-constrained\nversion where amount of available resources is decided taking into account the minimal\namount of resources needed to solve the instance.\n\nDescription of the original domain: This one was for the numeric track and proved to be a\nvery tough resource management domain. Several interesting issues in encoding arise as\nwell as the subsequent problem of planning with the domain. In particular, resources can\nbe combined to construct vehicles of various kinds. Since these vehicles are not available\ninitially, this is an example of a problem in which new objects are created. PDDL does not\nconveniently support this concept at present, so it is necessary to name "potential"\nvehicles at the outset, which can be realised through construction. A very high degree of\nredundant symmetry exists between these "potential" vehicles, since it does not matter\nwhich vehicle names are actually used for the vehicles that are realised in a\nplan. Planners that begin by grounding all actions can be swamped by the large numbers of\npotential actions involving these potential vehicles, which could be realised as one of\nseveral different types of actual vehicles.\n\nPlan quality is judged by a linear combination of labour use, pollution creation and\nresource consumption. There is scope for constructing very hard metrics that involve\nmaximising housing construction subject to an increasing pollution penalty (say), to\nensure that optimal plan quality is bounded.',
 'ipc': '2018',
 'name': 'settlers',
 'problems': [('settlers-opt18/domain.pddl', 'settlers-opt18/p11.pddl'),
              ('settlers-opt18/domain.pddl', 'settlers-opt18/p07.pddl'),
              ('settlers-opt18/domain.pddl', 'settlers-opt18/p06.pddl'),
              ('settlers-opt18/domain.pddl', 'settlers-opt18/p10.pddl'),
              ('settlers-opt18/domain.pddl', 'settlers-opt18/p17.pddl'),
              ('settlers-opt18/domain.pddl', 'settlers-opt18/p01.pddl'),
              ('settlers-opt18/domain.pddl', 'settlers-opt18/p16.pddl'),
              ('settlers-opt18/domain.pddl', 'settlers-opt18/p20.pddl'),
              ('settlers-opt18/domain.pddl', 'settlers-opt18/p19.pddl'),
              ('settlers-opt18/domain.pddl', 'settlers-opt18/p03.pddl'),
              ('settlers-opt18/domain.pddl', 'settlers-opt18/p15.pddl'),
              ('settlers-opt18/domain.pddl', 'settlers-opt18/p14.pddl'),
              ('settlers-opt18/domain.pddl', 'settlers-opt18/p02.pddl'),
              ('settlers-opt18/domain.pddl', 'settlers-opt18/p18.pddl'),
              ('settlers-opt18/domain.pddl', 'settlers-opt18/p05.pddl'),
              ('settlers-opt18/domain.pddl', 'settlers-opt18/p13.pddl'),
              ('settlers-opt18/domain.pddl', 'settlers-opt18/p09.pddl'),
              ('settlers-opt18/domain.pddl', 'settlers-opt18/p08.pddl'),
              ('settlers-opt18/domain.pddl', 'settlers-opt18/p12.pddl'),
              ('settlers-opt18/domain.pddl', 'settlers-opt18/p04.pddl')]}]
