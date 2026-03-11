# me133b

Do:
- [X] Change from probability to logits
- [ ] (Not as important) Make the grid finer / how to actually do walls rather than xs as we make the grid finer?
- [X] Show path on the visualisation
- [X] Have unknown cost be higher than known cost (D* can be optimistic as to there not being walls)
- [ ] Include the cost of pushing through fire
    - [X] Add it to D* Lite planner
    - [ ] Discuss heuristic (see * below)
- [ ] Possibly visualising real vs apparent (ie robot has no clue of how fire is outside its local area, and when added with walls currently it becomes hard to visualise what is real and what is perception)
- [ ] Fix sensor (currently there is a probability that it doesn’t see the wall, and this means that it will keep going and detects a far away wall, whereas what we want is just a ± a couple grid elements worth of uncertainty, the wall isn’t invisible)
- [ ] Have it go to a goal and return, not just go and stay

Discuss:
- [ ] How fire should start (small and rapid expansion vs large and slow expansion)
- [ ] What metric we're optimising for
- [ ] If it hits the wall, should it take some sort of extra penalty (or just that that action is 1 step wasted)?
- [ ] * How we should couple our D* costs with the real ones? (Discuss with Günter?)

Later:
- [ ] Actually change comments so it reflects what goes on in the code (mb yall)
