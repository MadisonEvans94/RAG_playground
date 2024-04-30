> *created on 2024-04-25

#evergreen1 

---
**links**: 
**brain-dump**: 

---
## 1NF, 2NF, and 3NF

### Formal Definitions 

**First Normal Form (1NF)**:
A relation $R$ is in First Normal Form if and only if all underlying domains contain atomic values only. There are no repeating groups or arrays in any of the relation's tuples.

**Second Normal Form (2NF)**:
A relation $R$ is in Second Normal Form if and only if it is in $1NF$ and every non-prime attribute is fully functionally dependent on the primary key of the relation. This means that there is no partial dependency of any column on the primary key.

**Third Normal Form (3NF)**:
A relation $R$ is in Third Normal Form if and only if it is in $2NF$ and every non-prime attribute is non-transitively dependent on the primary key. In other words, there should be no transitive dependency for non-prime attributes on the primary key. This means a non-prime attribute should not depend on another non-prime attribute.

### Informal Definitions 

**First Normal Form (1NF):** *Atomic columns only*
>Ensure that all columns contain atomic, indivisible values with no repeating groups or arrays.

**Second Normal Form (2NF):** *Whole key dependency*
>Ensure that all non-key attributes are fully functionally dependent on the entire primary key (applies only to tables with composite primary keys).

3. **Third Normal Form (3NF):** *No transitives, please*
>Ensure there are no transitive dependencies; every non-key attribute must depend directly on the primary key and not through another non-key attribute.


**When determining the primary key for any relation, consider this:**
1. Look for the minimal set of attributes that can uniquely identify a row in the table. These attributes should be such that no subset of them can be a primary key.
2. Check if any functional dependencies imply that one or more attributes uniquely determine all other attributes in the table.
3. Ensure that the primary key doesn’t have any redundant information, which means that every part of the key is necessary for uniqueness.

---
## Boyce-Codd Normal Form 

**Candidate Key**: an attribute (or combination of attributes) that uniquely identifies a row in the table 
**Prime Attribute**: An attribute that belongs to at least one candidate key
**Non-prime Attribute**: An attribute that doesn't belong to any candidate key 

2NF says we can't have a non-prime attribute that depends on part of a candidate key 

| Locker_ID | Reservation_Start_Date | Reservation_End_Date | Reservation_End_Day |
| --------- | ---------------------- | -------------------- | ------------------- |
| 221       | 14-May-2019            | 12-Jun-2019          | Wednesday           |
| 303       | 07-Jun-2019            | 12-Jun-2019          | Wednesday           |
| 537       | 14-May-2019            | 17-May-2019          | Friday              |
in the above example, our candidate keys are as follows: 

1) { Locker_ID, Reservation_Start_Date }
2) { Locker_ID, Reservation_End_Date }

Both candidate keys can uniquely identify the entire row, however, one of these candidate keys violates 2NF, because if we chose option 2, then Reservation_End_Day now depends on a portion of the candidate key instead of the full candidate key. 

*Therefore*, the example is **not** in Boyce-Codd Normal Form 

---

## Armstrong's Axioms 
**Reflexive Rule**: If X ⊇ Y, then X →Y.  

Armstrong's Reflexive Rule states that if you have a set of attributes X, and Y is a subset of X (meaning Y contains only attributes that are also in X), then X functionally determines Y. In plain English, if you have all the information in X, you naturally have the information in Y since Y is just part of X.

>Imagine you have a set of attributes X = {StudentID, StudentName, EnrollmentDate} and Y = {StudentID, StudentName}. Since Y is just a part of X, knowing a student's ID and name (X), you automatically know their ID and name (Y). So X functionally determines Y.

**Augmentation Rule**: {X → Y} |=XZ → YZ. 

The Augmentation Rule says that if you know a set of attributes X determines another set Y, then adding another attribute Z to both X and Y doesn't change that relationship—X plus Z still determines Y plus Z. It's like saying if knowing someone's social security number lets you retrieve their tax record, then knowing their social security number and favorite color still lets you retrieve their tax record and favorite color.

>For example, if knowing a StudentID (X) lets you determine a student's EnrollmentDate (Y), then knowing the StudentID and the student's Major (XZ) will still let you determine the student's EnrollmentDate and Major (YZ).

**Transitive Rule**: {X → Y, Y → Z} |=X → Z.

The Transitive Rule means if a set of attributes X determines another set Y, and Y in turn determines a third set Z, then you can say that X determines Z as well. It's like a chain reaction: if you know the first link leads to the second, and the second leads to the third, then the first must lead to the third.

>For example, if a student's ID (X) determines their login name (Y), and their login name determines their email address (Z), then the student's ID also determines their email address.


---

## Closures

if you can determine attribute closure then you can determine candidate keys. Let's say we have the following: 

```
R(A, B, C, D, E)
FD: { A -> B, B -> C, C -> D, D -> E}
```

Due to **transitive rule**, we can say `A -> C`
Due to **reflexive rule**, we can say `A -> A`
Due to **union property**, we can say `A -> BCDE`
...however, `B` **cannot** determine `A`

We can define a **closure** as the *set* of attributes that are determined by an attribute `x`. 

So in the case of the example above...
- `ABCDE` is the closure of `A`
- `BCDE` is the closure of `B`
- `CDE` is the closure of `C`
- `DE` is the closure of `D`
- `E` is the closure of `E`

---

## Lossless Decomposition

**Lossless decomposition** refers to breaking down a database schema into two or more schemas (relations) in such a way that the original schema can be perfectly reconstructed by joining the decomposed schemas. It ensures that no information is lost in the process of decomposition.

To determine if a decomposition is lossless, you use the following condition:

For a decomposition of a relation $R$ into two relations $R1$ and $R2$ to be lossless, at least one of the following two conditions should hold true:
1. The intersection of $R1$ and $R2$'s attributes ($R1 ∩ R2$) contains a candidate key for either $R1$ or $R2$.
2. The functional dependencies in R imply that ($R1 ∩ R2$) functionally determines $R1$ or $R2$.

If one of these conditions holds, the original relation R can be reconstructed without any loss of information by performing a natural join of $R1$ and $R2$. If neither condition is met, the decomposition may be lossy, meaning that some information might be lost or spurious tuples may be generated when the relations are joined back together.

### Example 

Let's take a simple example to illustrate lossless decomposition with an easy-to-understand table setup. Suppose we have a relation \( R \) with the following schema and tuples:

**Relation R (StudentID, CourseID, Grade)**

| StudentID | CourseID | Grade |
|-----------|----------|-------|
| 001       | Math101  | A     |
| 002       | Eng202   | B     |
| 003       | Hist303  | A     |

We want to decompose \( R \) into two relations, \( R1 \) and \( R2 \), as follows:

**Relation R1 (StudentID, CourseID)**

| StudentID | CourseID |
|-----------|----------|
| 001       | Math101  |
| 002       | Eng202   |
| 003       | Hist303  |

**Relation R2 (StudentID, Grade)**

| StudentID | Grade |
|-----------|-------|
| 001       | A     |
| 002       | B     |
| 003       | A     |

### Checking for Lossless Join

To check if this decomposition is lossless, we need to examine the intersection of \( R1 \) and \( R2 \) and apply the conditions for lossless join.

- The intersection of \( R1 \) and \( R2 \) is **{StudentID}**.
- Suppose the functional dependency \( StudentID \rightarrow CourseID, Grade \) holds across the whole database.

#### Condition Check
- \( R1 \cap R2 = \{StudentID\} \) does not functionally determine all of \( R1 \) or all of \( R2 \) based on the original dependencies unless \( StudentID \) alone was enough to determine both CourseID and Grade (which it does not in this specific scenario unless stipulated). 
- However, we can state that if \( StudentID \) functionally determines \( CourseID \) and \( Grade \), then the decomposition is lossless because the intersection \( \{StudentID\} \) is a key for the original table.

### Joining Back the Relations
Performing a natural join on \( R1 \) and \( R2 \) based on \( StudentID \):

| StudentID | CourseID | Grade |
|-----------|----------|-------|
| 001       | Math101  | A     |
| 002       | Eng202   | B     |
| 003       | Hist303  | A     |

The joined table matches the original table \( R \) exactly, indicating that no information is lost in this decomposition.

This example shows how decomposition is lossless if you can ensure that all data from the original relation can be perfectly reconstructed by a natural join of the decomposed relations, especially when the intersection of the decomposed relations (based on common attributes) can uniquely identify the attributes in each decomposed relation.