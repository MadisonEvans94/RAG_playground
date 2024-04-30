
###### Upstream: [[Software Development]]
###### Siblings: [[SOLID]]
#evergreen1 

### Origin of Thought:
- n/a

### Underlying Question: 
- how are these terms defined and what are their differences and similarities? 


### Solution/Reasoning: 

**Imperative Programming**

- Imperative programming is a paradigm that uses statements to change a program's state 
- It's all about how to achieve the desired result, and It involves giving the computer a sequence of tasks which are then executed in order 
- If you've ever written a for-loop, an if-else statement, or a switch-case statement, you've done imperative programming

Consider the following example in JavaScript, where we want to double all numbers in an array:

```js
let numbers = [1, 2, 3, 4, 5]; let doubled = [];  

for(let i = 0; i < numbers.length; i++) {     
	doubled.push(numbers[i] * 2); 
}
```

In this example, we manually control how the iteration over the array is performed and push each doubled number into the new array.

**Declarative Programming**

- Declarative programming is a higher-level programming paradigm that involves providing the desired result without explicitly listing out the steps that need to be taken to achieve the result
- It's more about what to do and what the desired outcome should be, and less about how to achieve it.

Here's how we might double all numbers in an array in a declarative style:


```js
let numbers = [1, 2, 3, 4, 5]; 
let doubled = numbers.map(num => num * 2);
```


In this example, we're not manually controlling the iteration over the array. Instead, we're defining what we want: a new array (`doubled`), where each number is the double of the corresponding number in the `numbers` array.

**Comparison**

Imperative code tends to be "lower-level" in that it's closer to the way machines actually operate. It's more verbose and usually more flexible. It can also be easier to understand, since it's essentially a detailed set of instructions for the computer.

Declarative code is typically "higher-level", more abstracted, and more succinct. It can make code easier to write and read, but it might also make it harder to understand how it works, especially for beginners. Declarative code also tends to be less flexible, as it relies on the underlying functions or libraries to decide how to perform tasks.

SQL is an example of a declarative language: you specify _what_ data you want, but not _how_ to retrieve it. On the other hand, most popular programming languages like JavaScript, Python, and Java are imperative, although they have declarative features or can be used in a declarative style. HTML and CSS are also examples of declarative languages.

In summary, whether you use an imperative or declarative style might depend on the particular task at hand, the specific tools you're using, and your personal preference or comfort level with these paradigms.

### Examples (if any): 


