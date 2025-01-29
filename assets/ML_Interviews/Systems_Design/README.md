# Systems Design

1. [Computer Architecture](#computer-architecture)
2. [Application Architecture](#application-architecture)




### Recall: bits and bytes

- a bit is the **smallest** unit of a computer: `0` or `1`
- a byte is **8 bits**. ex. `0 1 0 1 0 1 0 1`



## Computer Architecture

How does your computer store and transfer information?

### Disk

A disk is the primary storage device in a computer. It is **persistent**, meaning the data is persisted regardless of the state of computer (turned on or off).


### RAM

Random Access Memory is also used to store information, but it is smaller in size compared to disk drives. RAM is **significantly faster to read and write to than disk**.


RAM keeps applications you have open in memory, including any `variables` your program has allocated. It is volatile, meaning that data gets erased when the computer is shut down. That's why you should save your work!



*The RAM and disk do not directly communicate or facilitate data transfers between each other. This is what the CPU does.*


### CPU

The CPU is the intermediary between RAM and disk: it reads/writes from the RAM and disk(s).

For example, when you write code and run it, your code is translated into a set of binary instructions stored in RAM. In other words, the CPU reads and executes these instructions, which may involve manipulating data stored elsewhere in RAM or on disk. An example of reading from disk, would be opening a file in your file system and reading it line-by-line.

All computations are done within the CPU, such as addition/subtraction/multiplication etc. This occurs in a matter of milliseconds. It fetches instructions from the RAM, decodes those instructions and executes the decoded instructions. On the lowest level, all these instructions are represented as bytes.

The CPU also consists of a cache. A cache is extremely fast memory that lies on the same die as the CPU.


### Cache

Most CPUs have cache: physical components that are much faster than RAM, but store less data.


Whenever a read operation is requested, the cache is checked before the RAM and the disk. If the data requested is in the cashe, and is unchanged since the last time it was accessed, it will be fetched form the cache (not the RAM). It is up to the operating system to decide what gets stored in the cache.

Caching is an important concept applied in many areas beyond computer architecture. For example, web browsers use cache to keep track of frequently accessed web pages to load them faster. This stored data might include HTML, CSS, JavaScript, and images, among other things. If the data is still valid from the time the page was cached, it will load faster. But in this case, the browser is using the disk as cache, because making internet requests is a lot slower than reading from disk.


### Moore's Law

<p># of transistors = proxy for CPU speed

**Moore's Law**: The number of transistors doubles while costs halve every two years. Exponential growth.



## Application Architecture