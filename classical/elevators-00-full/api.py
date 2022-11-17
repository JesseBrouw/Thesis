domains = [
{'description': 'The scenario is the following: There is a building with N+1 floors, numbered from 0 to N. The building can be separated in blocks of size M+1, where M divides N. Adjacent blocks have a common floor. For example, suppose N=12 and M=4, then we have 13 floors in total (ranging from 0 to 12), which form 3 blocks of 5 floors each, being 0 to 4, 4 to 8 and 8 to 12. The building has K fast (accelarating) elevators that stop only in floors that are multiple of M/2 (so M has to be an even number). Each fast elevator has a capacity of X persons. Furthermore, within each block, there are L slow elevators, that stop at every floor of the block. Each slow elevator has a capacity of Y persons (usually Y<X). There are costs associated with each elavator starting/stoping and moving. In particular, fast (accelarating) elevators have negligible cost of starting/stoping but have significant cost while moving. On the other hand, slow (constant speed) elevators have significant cost when starting/stoping and negligible cost while moving. Travelling times between floors are given for any type of elevator, taking into account the constant speed of the slow elevators and the constant acceleration of the fast elevators. There are several passengers, for which their current location (i.e. the floor they are) and their destination are given. The planning problem is to find a plan that moves the passengers to their destinations while minimizing the total cost of moving the passengers to their destinations . The total cost is increased each time an elevator starts/stops or moves.',
 'ipc': '2000',
 'name': 'elevators',
 'problems': [('elevators-00-full/domain.pddl',
               'elevators-00-full/f1-0.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f1-1.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f1-2.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f1-3.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f1-4.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f10-0.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f10-1.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f10-2.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f10-3.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f10-4.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f11-0.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f11-1.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f11-2.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f11-3.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f11-4.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f12-0.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f12-1.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f12-2.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f12-3.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f12-4.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f13-0.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f13-1.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f13-2.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f13-3.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f13-4.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f14-0.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f14-1.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f14-2.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f14-3.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f14-4.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f15-0.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f15-1.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f15-2.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f15-3.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f15-4.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f16-0.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f16-1.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f16-2.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f16-3.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f16-4.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f17-0.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f17-1.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f17-2.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f17-3.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f17-4.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f18-0.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f18-1.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f18-2.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f18-3.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f18-4.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f19-0.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f19-1.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f19-2.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f19-3.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f19-4.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f2-0.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f2-1.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f2-2.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f2-3.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f2-4.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f20-0.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f20-1.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f20-2.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f20-3.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f20-4.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f21-0.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f21-1.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f21-2.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f21-3.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f21-4.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f22-0.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f22-1.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f22-2.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f22-3.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f22-4.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f23-0.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f23-1.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f23-2.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f23-3.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f23-4.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f24-0.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f24-1.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f24-2.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f24-3.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f24-4.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f25-0.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f25-1.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f25-2.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f25-3.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f25-4.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f26-0.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f26-1.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f26-2.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f26-3.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f26-4.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f27-0.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f27-1.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f27-2.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f27-3.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f27-4.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f28-0.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f28-1.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f28-2.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f28-3.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f28-4.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f29-0.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f29-1.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f29-2.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f29-3.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f29-4.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f3-0.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f3-1.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f3-2.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f3-3.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f3-4.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f30-0.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f30-1.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f30-2.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f30-3.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f30-4.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f4-0.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f4-1.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f4-2.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f4-3.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f4-4.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f5-0.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f5-1.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f5-2.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f5-3.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f5-4.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f6-0.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f6-1.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f6-2.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f6-3.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f6-4.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f7-0.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f7-1.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f7-2.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f7-3.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f7-4.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f8-0.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f8-1.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f8-2.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f8-3.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f8-4.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f9-0.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f9-1.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f9-2.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f9-3.pddl'),
              ('elevators-00-full/domain.pddl',
               'elevators-00-full/f9-4.pddl')]}
]