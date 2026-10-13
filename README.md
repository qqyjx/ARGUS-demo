# ARGUS demo: argumentation-based minimal-change repair

A minimal demo for the paper *ARGUS: Argumentation-Based Minimal-Change Repair for Verifiable LLM Self-Explanations* (UAI 2026).

**Notice.** This is a minimal demo that illustrates the method. It is **not** the code used for the paper's experiments, and it does not reproduce any number reported in the paper. All inputs are small synthetic toy cases, and the LLM is a deterministic rule-based mock. The demo prints only what it computes on those toy inputs.

## The paper in one paragraph

Natural-language rationales from LLMs are often unfaithful, and there is no principled way to fix them when new evidence arrives. ARGUS turns a rationale into a Dung-style abstract argumentation framework (arguments and attacks) and verifies that the target argument, which carries the answer, is accepted under grounded or preferred semantics. If new evidence makes the target unaccepted, ARGUS computes a minimum-cost set of edit operations (`add_arg`, `del_arg`, `add_att`, `del_att`) that restores the desired status. This is the minimal-change principle of AGM belief revision. The cost can be uniform, confidence-weighted, or structure-preserving (deletions cost more than additions). The paper encodes the search in answer set programming; this demo uses exact enumeration.

## Quickstart

Python 3.10 or newer. The demo uses only the standard library and needs no network or API key. `pytest` is needed for the tests.

```bash
python -m argus.demo                 # all three toy cases
python -m argus.demo --case museum   # one case: diagnosis | museum | capital
python -m argus.demo --semantics pr  # repair under preferred semantics
pip install pytest && pytest -q
```

## What the demo shows

For each case the demo prints the extracted arguments, attacks and support links, the extensions and accepted arguments under grounded (`gr`) and preferred (`pr`) semantics, the verification verdict for the target, and (after an evidence update) the optimal repair under three cost models.

| Toy case | Shows |
|---|---|
| `diagnosis` | A late lab result removes the defence of the target. Verification fails. Optimal repairs differ by cost model: an attack removal under uniform cost, an added counterargument under structure-preserving cost. |
| `museum` | Two conflicting sources attack each other. The target is `undecided` under grounded semantics but accepted under preferred semantics. |
| `capital` | The explanation is already consistent and the evidence is irrelevant. The repair is empty at cost 0 (vacuity). |

| Demo component | File | Paper |
|---|---|---|
| Argumentation framework, grounded/preferred/stable semantics, defense sets | `argus/af.py` | Sec. 2 (Definitions of AF, defense set) |
| Structured extraction, relation discovery (mock LLM and NLI) | `argus/llm.py`, `argus/pipeline.py` | Sec. 3.1, 3.2 |
| Acceptability check: accepted / rejected / undecided | `argus/verify.py` | Sec. 3.3, Explanation Verification Task |
| Evidence update, edit operations, cost models | `argus/ops.py` | Sec. 2 (Minimal-Change Repair Problem), Sec. 3.4 |
| Optimal repair (exact search, k-neighbourhood filter, budget) | `argus/repair.py` | Sec. 3.4, Algorithm 1 |
| Toy cases | `argus/cases.py` | Running example (Sec. 1) in `diagnosis` |

"Verifiable" means this: the explanation is turned into an argumentation framework, and the answer's argument must be accepted under the chosen semantics. For an accepted target the demo also reports its defense sets as a certificate.

## Repository layout

```
argus/
  af.py        AF, semantics (gr/pr/st), acceptance status, defense sets
  ops.py       Op, EvidenceUpdate, apply, CostModel
  verify.py    verification verdict and certificate
  repair.py    exact minimum-cost repair search
  llm.py       LLMClient protocol, MockLLM, ToyKB
  pipeline.py  extraction -> AF, evidence -> Delta, repair of a graph
  cases.py     three synthetic toy cases
  demo.py      python -m argus.demo
tests/         semantics, verification, operations, repair minimality, pipeline
```

## Tests

- `test_semantics.py`: textbook frameworks (chain, mutual attack, odd cycle, a framework whose preferred extensions differ from the grounded one, self-attack) and the defense set of the running example.
- `test_verify.py`: accepted, rejected and undecided verdicts.
- `test_repair.py`: known optimum of the running example under each cost model, and a comparison of the search against an exhaustive reference (on that example and on random small frameworks).
- `test_pipeline.py`: the mock LLM, the three toy cases and the demo command.

## Simplifications (differences from the paper)

- **Mock LLM.** Extraction splits sentences and "X because Y", and takes confidence from hedge words. The NLI model and the attack template library are replaced by a directed phrase lookup in a small synthetic knowledge base. There is no JSON schema validator and no re-prompting. The mock understands only the toy cases. `LLMClient` is the place to plug in a real model; no real client is included.
- **Supports.** Support links (from premises and entailment rules) are recorded but are not part of the attack relation, as in the paper. They are not used in any computation.
- **Search.** Exact enumeration of operation sets by increasing size replaces the ASP encoding. It is suitable only for toy sizes. The beam-search fallback is not implemented. The k-neighbourhood is a filter on candidate operations, and `max_ops` is the budget; if nothing is found the result is flagged unrepairable within budget, and the paper's regeneration fallback is not implemented.
- **Candidate operations.** New arguments come from counterarguments proposed by the mock. A new attack is admitted only if the mock NLI labels it a contradiction, so a repair cannot add an unsupported attack between existing arguments. The search itself accepts any candidate set (the unit tests use all pairs). The target is never deleted.
- **Status.** Under preferred and stable semantics, "in" means credulously accepted (member of at least one extension) and "out" means not credulously accepted. `undecided` arises only under grounded semantics.
- **Certificate.** The demo returns defense sets when the target is accepted, and lists the target's attackers otherwise. The paper returns a defense set in both cases.
- **Tie-breaking.** The paper leaves ties to domain preferences. The demo picks, among minimum-cost repairs, the one with fewer argument deletions, then fewer deletions, then fewer operations, then lexicographic order. All optimal sets are returned and counted.
- **Evidence.** The pipeline builds updates that only add arguments and attacks, one argument per evidence sentence. `EvidenceUpdate` supports removals and the tests use additions only.
- **Theory.** The AGM postulates (success, inclusion, vacuity) are checked on examples, not proved. Complexity results are not exercised.

## Citation

```bibtex
@inproceedings{xiao2026argus,
  title     = {{ARGUS}: Argumentation-Based Minimal-Change Repair for Verifiable {LLM} Self-Explanations},
  author    = {Xiao, Yifan and Li, Shijie},
  booktitle = {Proceedings of the 42nd Conference on Uncertainty in Artificial Intelligence (UAI 2026)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {245},
  year      = {2026},
  publisher = {PMLR}
}
```

## License

MIT. See `LICENSE`.
