# Prompt

> if `<sep>` is not available, use "\n"

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
