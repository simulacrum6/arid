The following resource contains 29,732 entailment rules between typed predicates. Rules were learned over ~10,000 typed predicates, where types are taken from a dictionary that contains ~150 types. This is the same dataset that was used by Schoenmackers et al. in their paper: "Horn-clause inference rules learned by Sherlock". A mapping of 1.1 million instances to the 150 types is provided at the website:

http://www.cs.washington.edu/research/sherlock-hornclauses/

The file is at:

http://www.cs.washington.edu/research/sherlock-hornclauses/allclassinstances.txt.gz

The rules were learned by the algorithm described in the Berant et al.'s paper: "Global Learning of Typed Entailment Rules".
The rules were learned with the parameter value \log \eta = -0.6.

The file is a tab-delimited text file with 3 columns:

1. Entailing typed predicate description
2. Edge type description
3. Entailed typed predicates desciprtion

A typed predicates is described by a string of the following form "<predicate::type1::type2>". For example, the typed predicate "<live in::animal::region>" describes the predicate "live in" where the first argument is of the type "animal" and the second argument is of the type "region".

An edge type description specificies whether an edge is a "direct-mapping edge" or "reversed-mapping edge". Direct-mapping edges are described by the string "->" and reversed-mapping edges by the string "-R>". No other strings are valid.

Rules were learned over the same set of typed predicates used by Schoenmackers et al. in their paper:
"Horn-clause inference rules learned by Sherlock"




