# IDRR_data

Init data: 
    pdtb2.p1.csv
    pdtb3.p1.csv
    conll_test.p1.csv
    conll_blind_test.p1.csv

## Columns

* init columns:
  'arg1', 'arg2', 'conn1', 'conn2',
  'conn1sense1', 'conn1sense2', 'conn2sense1', 'conn2sense2',
  'relation', 'split'
* process:
  connX -> ans_wordX, ans_wordXid
  connXsenseY -> labelXY, labelXYid
  relation: filter
  split: filter
* processed columns:
  'index', 'arg1', 'arg2', 'conn1', 'conn2',
  'conn1sense1', 'conn1sense2', 'conn2sense1', 'conn2sense2',
  'relation', 'split',
  'label11', 'label11id', 'label12', 'label12id',
  'label21', 'label21id', 'label22', 'label22id',
  'ans_word1', 'ans_word1id', 'ans_word2', 'ans_word2id'

# Prompt

> if <sep> is not available, use "\n"

## Default

* plain

```txt
{arg1}<sep>{arg2}
```

{arg1}`<sep>`{arg2}

* base

```txt
Argument 1:
{arg1}

Argument 2:
{arg2}

Question: What is the discourse relation between Argument 1 and Argument 2?
A. Comparison
B. Contingency
C. Expansion
D. Temporal

Answer:
```

Argument 1:\n{arg1}\n\nArgument 2:\n{arg2}\n\nQuestion: What is the discourse relation between Argument 1 and Argument 2?\nA. Comparison\nB. Contingency\nC. Expansion\nD. Temporal\n\nAnswer:

## Connective-based

* 

## subtext

* plain

```txt
{subtext}
```

{subtext}

* base: subtext replace arguments

```txt
Implicit meaning:
{subtext}

Question: Based on the implicit meaning, what is the discourse relation between arguments?
A. Comparison
B. Contingency
C. Expansion
D. Temporal

Answer:
```

Implicit meaning:\n{subtext}\n\nQuestion: Based on the implicit meaning, what is the discourse relation between arguments?\nA. Comparison\nB. Contingency\nC. Expansion\nD. Temporal\n\nAnswer:

* base: fuse subtext and arguments with the prompt

```txt
Argument 1:
{arg1}

Argument 2:
{arg2}

Implicit meaning:
{subtext}

Question: Based on the implicit meaning, what is the discourse relation between Argument 1 and Argument 2?
A. Comparison
B. Contingency
C. Expansion
D. Temporal

Answer:
```

Argument 1:\n{arg1}\n\nArgument 2:\n{arg2}\n\nImplicit meaning:\n{subtext}\n\nQuestion: Based on the implicit meaning, what is the discourse relation between Argument 1 and Argument 2?\nA. Comparison\nB. Contingency\nC. Expansion\nD. Temporal\n\nAnswer:
