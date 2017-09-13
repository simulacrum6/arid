# Entailment Graph Resources

This folder contains all resources, required to run the `arid.inference.classifiers.EntailmentGraph` classifier:

1. Dictionary to map instances to types, `class-instance-mapping.txt`
2. Edgelist of the entailment graph, `original-edgelist.txt`, `entailment-graph.json`, `entailment-graph_tidy.csv`

The JSON-Edgelist follows the `analysis` format for datasets of this project, the CSV-Edgelist follows the `tidy` format. 

To use these resources, it is recommended to load them via the `arid.utils.resources` module. It will return the data in the correct format for the classifier.
The following contains the original readme file, provided with the datasets:

## Original readme Levy 2014

The following resource[`original-edgelist.txt`] contains 29,732 entailment rules between typed predicates. Rules were learned over ~10,000 typed predicates, where types are taken from a dictionary[`class-instance-mapping.txt`] that contains ~150 types. This is the same dataset that was used by Schoenmackers et al. in their paper: "Horn-clause inference rules learned by Sherlock". A mapping of 1.1 million instances to the 150 types is provided at the website:

http://www.cs.washington.edu/research/sherlock-hornclauses/

The file is at:

http://www.cs.washington.edu/research/sherlock-hornclauses/allclassinstances.txt.gz

The rules were learned by the algorithm described in the Berant et al.'s paper: "Global Learning of Typed Entailment Rules".
The rules were learned with the parameter value \log \eta = -0.6.

The file is a tab-delimited text file with 3 columns:

1. Entailing typed predicate description
2. Edge type description
3. Entailed typed predicates desciprtion

A typed predicates is described by a string of the following form `<predicate::type1::type2>`. For example, the typed predicate `<live in::animal::region>` describes the predicate "live in" where the first argument is of the type "animal" and the second argument is of the type "region".

An edge type description specificies whether an edge is a "direct-mapping edge" or "reversed-mapping edge". Direct-mapping edges are described by the string "->" and reversed-mapping edges by the string "-R>". No other strings are valid.

Rules were learned over the same set of typed predicates used by Schoenmackers et al. in their paper:
"Horn-clause inference rules learned by Sherlock"

## Original Readme Berant 2010

What is this?
============
This is a resource of predicative entailment rules created according to the method described in Chapter 5 of the thesis draft "Global Learning of Textual Entailment Graphs" provided with this package. 

What does it contain?
====================
The package contains 3 resources in two formats - one syntactic and one lexical.

What to use?
===========
The default is:
if you want to work over a non-syntactic representation of the rules - use the file "reverb_local_clsf_all.txt". Use the default score threshold: '0'.

if you want to work over a syntactic representation of the rule - use the tables "reverb_local_distlexfeatures_templates" and "reverb_local_distlexfeatures_rules" that are in the scheme "reverb" provided as an sql dump with this package.

Files in this package:
======================
1. readme.txt - this file

2. berant_thesis.pdf - draft of thesis describing the algorithm used for creating the resource

3. reverb_clsf_all.txt - resource created by applying the local algorithm clsf_all (Section 5.2.2.1) over the REVERB data set in a shallow representation.  

4. reverb_global_clsf_all_htl_lambda0.05.txt - resource created by applying the global algorithm HTL (Section 5.2.2.2) over the REVERB data set in a shallow representation with parameter \lambda=0.05. Number of rules: 

5. reverb_global_clsf_all_tncf_lambda_0.1.txt - resource created by applying the global algorithm TNCF (Section 5.2.2.2) over the REVERB data set in a shallow representation with parameter \lambda=0.1.

6. reverb_syntax.sql - all resources above (3-5) in a syntactic representation exported from a MySQL database.

reverb_clsf_all.txt format
==========================
contains 52,164,051 lines.
Tab-delimited file with three columns

1. Description of entailing predicate
2. Description of entailed predicate.
3. Classifier score - between -infinity and +infinity. User should choose a score threhold. Recommended threhold: '0'.

Description is simply a string optinally followed by "@R@", which denotes that the first argument is "Y" and the second argument is "X".
For example, the string "abandon" corresponds to the predicate "X abandon Y", while the string "abandon@R@" corresponds to the predicate "Y abandon X". Therefore, the rule "be abandoned by-->abandon@R@" corresponds to "X be abandoned by Y-->Y abandon X".

A threhold score needs to be chosen and only rules that cross this threshold should be used. For example, a default threhold of '0' results in about 15 million rules.

IMPORTANT: Pairs of predicates with very low score (-10000.0) indicate that with high certainty there is not entailment.

reverb_global_clsf_all_htl_lambda0.05.txt and reverb_global_clsf_all_tncf_lambda_0.1.txt format
===============================================================================================
Output of global algorithms and therefore there are no scores, only the learned rules.
Tab-delimited files with 2 columns:
1. entailing predicate
2. entailed predicate

reverb database format
======================
Contains a syntactic representation (EastFirst parser) for the above three resources created by applying the conversion method from Section 5.2.3

1. reverb_clsf_all.txt - provided in the tables reverb_local_distlexfeatures_templates and reverb_local_distlexfeatures_rules 
2. reverb_global_clsf_all_htl_lambda0.05.txt - provided in the tables reverb_htl_distlexfeatures_templates and reverb_htl_distlexfeatures_rules 
3. reverb_global_clsf_all_tncf_lambda_0.1.txt - provided in the tables reverb_tncf_distlexfeatures_templates and reverb_tncf_distlexfeatures_rules 

template table format
---------------------
has four columns:

1. id - unique template id
2. unparsed - The original shallow representation as described above
3. full_description - The syntactic representation of the dependency path.
4. description - sometimes different shallow representations are mapped to the same syntactic representation. In this case we have different template IDs with the same syntactic description. We choose one of those as a "head". If a templateID is a "head" then description is equal to full_description. Otherswise, it is empty.

The syntactic description is a string according to Lin and Pantel's DIRT format and is simply a sequence of nodes and edges on the dependency path:
The first and last nodes are always noun variables denoted by "n"
edges denote the dependency direction and the dependency label, for instance "<nsubj<" or ">dobj>"
The other nodes provide the part-of-speech and lemma of the work for example the verb "affect" will appear as "v:affect:v" and the preposition "in" as "p:in:p"

rule table format
-----------------
has three columns:
1. left_element_id - entailed predicate template id (this is not a mistake!!!)
2. right_element_id - entailing predicate template id (this is not a mistake!!!)
3. score - bound between [0,1]

In the table for the local algorithm we only put rules for which the score > 0.5 so there are almost 10 million rules in this DB.

---------
website: jonatha6@post.tau.ac.il




