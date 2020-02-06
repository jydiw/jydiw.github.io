---
title: "Python3: List Comprehensions—Basics"
excerpt: "Post displaying the various ways of writing loops as list comprehensions in Python3."
# last_modified_at: 2018-01-03T09:45:06-05:00
header:
  teaser: "assets/images/markup-syntax-highlighting-teaser.jpg"
tags:
  - code
  - python
  - list_comprehensions
toc: true
---

List comprehensions are a compact, Pythonic way of writing `for` loops. In this series, we will learn how to write list comprehensions through a series of simple looping exercises. The solutions provided are not comprehensive (pun unintended), so take the time to come up with your own solutions or improve on the ones shown!

# Basic iterations
Let's take a look at the parts of a generic `for` loop:

```python
for {element} in {list}:
    {action}
```

A list comprehension has the same parts, just rearranged:

```python
[{action} for {element} in {list}]
```

## Exercise 1:
```python
'''
Determine whether the integers in a list are even (True) or odd (False).
'''

[1, 2, 3, 4]
>>>[False, True, False, True]
```
Essentially, we want to return the following list:
```python
>>>[(1 % 2 == 0), (2 % 2 == 0), (3 % 2 == 0), (4 % 2 == 0)]
```

This can be achieved using a `for` loop:

```python
even_list = []
for num in [1, 2, 3, 4]:
    even_list.append(num % 2 == 0)
print(even_list)
```

However, we can use list comprehension to write more compact code:

```python
even_list = [(num % 2 == 0) for num in [1, 2, 3, 4]]
print(even_list)
```

If you're still confused, see if this helps:
```python
[num for num in [1, 2, 3, 4]]
>>>[1, 2, 3, 4]

[(num % 2 == 0) for num in [1, 2, 3, 4]]
>>>[(1 % 2 == 0), (2 % 2 == 0), (3 % 2 == 0), (4 % 2 == 0)]
>>>[False, True, False, True]
```

## Exercise 2:
```python
'''
Convert all month names to their mmm abbreviations.
'''

months = [
  'January',
  'February',
  'March',
  'April',
  'May',
  'June',
  'July',
  'August',
  'September',
  'October',
  'November',
  'December'
  ]
>>>['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
```

`for` loop solution:
```python
mmm = []
for month in months:
    mmm.append(month[:3].lower())
print(mmm)
```

list comprehension solution:
```python
mmm = [month[:3].lower() for month in months]
print(mmm)
```

# Conditional iterations
## using `if` statements
If we only have an `if` conditional, it follows the `{list}` section in a list comprehension.
```python
[{action} for {element} in {list} if {condition}]
```
##
```python
'''
Define a function that replaces every letter of a string with its position in the alphabet.
(source: codewars)
'''

alphabet_position('hello 123')
>>>"8 5 12 12 15"
```
(one of many) `for` loop solutions:
```python
def alphabet_position(text):
    az = 'abcdefghijklmnopqrstuvwxyz'
    result = []
    for c in text.lower():
        if c.isalpha():
            result.append(str(az.index(c) + 1))
    return ' '.join(result)
```

the corresponding list comprehension solution:
```python
def alphabet_position(text):
    az = 'abcdefghijklmnopqrstuvwxyz'
    return ' '.join([str(az.index(c) + 1) for c in text.lower() if c.isalpha()])
```

## using `if`-`else` statements
However, if we have an `if`-`else` conditional, it follows the `{action}` section in a list comprehension.
```python
[{action1} if {condition1} else {action2} for {element} in {list}]
```
```python
'''
Write a function that scores a list of cards in Blackjack. Return the highest score of cards that is less than or equal to 21 or the lowest score of cards that is greater than 21.
(source: codewars)
'''

score_hand(['2', '8', 'J'])
>>>20

score_hand(['A', 'A'])
>>>12

score_hand(['A', 'J', '2'])
>>>23
```
(one of many) `for` loop solutions:
```python
def score_hand(cards):
    score = 0
    aces = cards.count('A')
    score += aces
    for card in cards:
        if card in 'JKQ':
            score += 10
        elif card.isnumeric():
            score += int(card)
    if (21 - score >= 10) and (aces > 0):
        score += 10
    return score
```

the corresponding list comprehension solution:
```python
def score_hand(cards):
    aces = cards.count('A')
    score = sum([10 if c in 'JQK' else int(c) for c in cards]) + aces
    if (21 - score >= 10) and (aces > 0):
        score += 10
    return score
```


## using multiple `if`-`else` iterations part 2
We can even incorporate multiple conditions into a list comprehension.
```python
[{action1} if {condition1} else {action2} if {condition2} else {action3} for {element} in {list}]
```


## Exercise 6: Nested iterations