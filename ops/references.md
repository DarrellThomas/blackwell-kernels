# References — prior literature for *The Cat and the Caretaker*

Companion to `paper_context_asymmetry.md`. The prior work the paper should engage. Three reasons it
matters: (1) some of these already named the variable and earned "fundamental" the hard way — copy
their method; (2) a couple supply the screening-off receipt the "fundamental" claim needs; (3) the
AI-oversight group **is** the inversion thesis, already a live research program. Read against these,
the novel residue (the *scoped equal*; "you can't tell which type from the inside") gets sharper and
the overreach gets sanded.

> **The bar to copy is Akerlof:** one model, one surprising prediction. That's how "asymmetry" became
> *fundamental* in economics — not by a pile of analogies.

**If you read three: Hayek, Holmström, the Off-Switch Game.**

---

## 1. Information asymmetry — the economics that already owns the word

- **Akerlof, "The Market for 'Lemons'" (1970)** — <https://www.sfu.ca/~allen/Ackerlof.pdf>
  The template. One toy model predicts a weird outcome (the market collapses to junk) that no rival
  variable predicts. This is the standard the "fundamental" claim has to clear.

- **Spence, "Job Market Signaling" (1973)** — orientation: <https://en.wikipedia.org/wiki/Signalling_(economics)>
  How the informed party signals across the gap. "Legible abstractions" is signaling under another name.

## 2. Principal–agent — the oversight problem, formalized 45 years ago

- **Holmström, "Moral Hazard and Observability" (1979)** — <https://gwern.net/doc/economics/1979-holmstrom.pdf>
  The thesis in math: the overseer sees outcomes, not the reasoning. The informativeness principle
  (reward on every signal carrying information about the hidden action) is exactly "design for
  empirical trust." The most likely precursor that pre-empts the paper — engage it head-on.

- **Coase, "The Nature of the Firm" (1937)** — <https://onlinelibrary.wiley.com/doi/full/10.1111/j.1468-0335.1937.tb00002.x>
  Why hierarchies exist at all — i.e., why there's a foreman/worker boundary to have asymmetry across.
  (Williamson's transaction-cost extension, orientation: <https://en.wikipedia.org/wiki/Transaction_cost>)

## 3. Distributed / local knowledge — the strongest counterweight to the modeling claim

- **Hayek, "The Use of Knowledge in Society" (1945)** — <https://www.econlib.org/library/Essays/hykKnw.html>
  The most important add. The "knowledge problem": the center **cannot** centralize the local, tacit
  knowledge held at the edge — not for lack of capacity, but in principle. Arms the "dignity of the
  scoped role" **and** directly attacks "the larger-context agent can model the smaller." Engaging this
  makes the paper honest about its biggest crack.

- **Scott, "Seeing Like a State" (1998)** — orientation: <https://en.wikipedia.org/wiki/Seeing_Like_a_State>
  Legibility as the powerful party's project, and how it fails when it flattens local knowledge.
  "Legible abstractions" with the politics put back in.

## 4. Epistemics of mutual modeling — what "A can model B" actually requires

- **Aumann, "Agreeing to Disagree" (1976)** — <https://www.princeton.edu/~bayesway/pu/Aumn.pdf>
  The formal machinery of what agents know about what others know (common knowledge). Forces precision
  on the hand-waved "bigger context ⇒ can model the other": modeling needs shared structure, not just
  more capacity.

## 5. Bounded rationality — the constraint, re-coined

- **Simon, "A Behavioral Model of Rational Choice" (1955)** — orientation: <https://en.wikipedia.org/wiki/Bounded_rationality>
  The "context window" is Simon's finite-information-processing constraint, 70 years on. Position the
  contribution as "bounded rationality with an explicitly variable, measurable bound."

## 6. AI oversight of a smarter agent — the inversion, already a research program

- **Amodei et al., "Concrete Problems in AI Safety" (2016)** — <https://arxiv.org/abs/1606.06565>
  Names "scalable oversight" — overseeing a system you can't fully evaluate. The inversion's home turf.

- **Irving, Christiano, Amodei, "AI Safety via Debate" (2018)** — <https://arxiv.org/abs/1805.00899>
  A mechanism for the smaller-context judge to oversee larger-context players. The PSPACE analogy is
  the rigorous version of "distributed oversight, no single human holds the full picture."

- **Hadfield-Menell, Dragan, Abbeel, Russell, "The Off-Switch Game" (2016)** — <https://arxiv.org/abs/1611.08219>
  Direct hit on §9.4 "preserve exit options." The off-switch / corrigibility problem, game-theoretically.
  Where the Oculus instinct already lives in the literature — go take it.

- **Burns et al. (OpenAI), "Weak-to-Strong Generalization" (2023)** — <https://arxiv.org/abs/2312.09390>
  The inversion as an **experiment**: can a weak (small-context) supervisor align a strong model?
  Empirical, falsifiable, the closest thing to the receipt you'd want. Pair with the factory's Type-2
  control and you have a real methods section.

- **Christiano et al., "Deep RL from Human Preferences" (2017)** — orientation: <https://arxiv.org/abs/1706.03741>
  The empirical-trust training loop in practice (humans judging outputs they can't fully reason about).

---

**How to use it.** §2 (Holmström) and §3 (Hayek) are the two that will most change the paper — one gives
the formal spine, the other forces honesty about the modeling claim. §6 is where the inversion stops
being speculation and becomes citable.

*Link note:* all verified live. Spence, Williamson, Scott, and Simon point to Wikipedia/orientation pages
(stable; originals sit behind JSTOR) rather than primary PDFs.
