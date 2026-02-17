---
layout: post
title: "what makes a good tool?"
date: 2026-02-16 10:00:00 +0000
categories: [Meta, Tools]
tags: [python, profiling, tools]
---
# what makes a good tool?

As someone writing code, we are constantly making tools. Over the years I’ve increasing come to the realisation that a lot of them have been rubbish. That sense of a codebase slowly getting away, that feeling of rather than using that lovingly created abstracted method I’ll just write the thing I need again for this one off. Maybe this is bad discipline, maybe it’s that I didn’t write things properly the first time, but there’s also something about making the right tool for the right job. 

There’s one thing to be clear about - anyone who’s ever built any kind of tooling (whether it’s a utility function, a super powerful fully abstracted multi-use library, or a really clever storage system for the spice rack in the kitchen), knows in their heart that if the alternative is either (a) better, or (b) easier - no system stands much chance against the unstoppable force of human laziness. Maybe that’s just me, I suspect it isn’t. 
There is a cost associated with remembering where things go, exactly what collection of commands do I run to do this operation, the friction that makes a simple task something that needs thought. That cost builds, and detracts from effort that could otherwise be spent solving the actually difficult part of the problem.


This brings me on to a little package I wrote. 
I can’t claim any invention here, this was entirely inspired by a friend of mine, and I thought it was such a good idea I wanted to code it up and make it available. So, thanks Alex! 
The basic point is this - profiling is powerful, but as with almost any high powered, multi functional, tool, it’s only good if you use it. 
Hence why time and time again when working out why our code is slow we end up resorting to some version of a print statement. It’s ugly, it’s only ever partly effective, and leaves me feeling somewhat depressed about failing the test of being a “proper programmer”.
This is really the guiding principle with the `gadget` package, make it easier to use than just printing, with less to remember, and more informative outputs. (And yes, I know I’m not the first person to have had this thought. The point isn’t that this solves profiling, it’s more a reflection on how I want my tools to work.)


For those interested - what does it actually do? In the minimal version the package can be pip installed (`pip install gadget-timer`) and imported as `from gadget import gadget`. Then wherever you need a little bit of profiling just add `gadget()`.
Then when the gadget line is found, we get a neat little print out with the time since the last gadget line, a summary of the line before the gadget call, and a formatted reference to the line in the file that I can click though on VSCode directly to the location.


![Gadget profiling example](/assets/img/gadget-example.png)
*Example from a tiny numpy ML script - we can see each stage including seeing reassuringly the backwards is substantially slower than forwards. We clearly don't get averages as with a real profiler, but we can spot major red flags.* 

That’s it. Everything I need. I can see exactly how long the interval took, I can scan down the left side of the output and visually see where the number is longer than the surrounding intervals, usually seeing the relevant line is enough to go, “oh yeah, obviously” but if not I can jump directly into the file via the reference. 

I’m sure some people will argue this isn’t proper profiling, and obviously that’s right, but for my uses 90% of the time the thing I really need is to quickly work out roughly where the time sinks are in my code. Which function did I think was quick enough that’s actually got a hidden $O(N^3)$ loop in it. 
And that’s where the point about good tools comes back, I could spin up a whole profiling setup and get vastly more information than I need to solve the problem I have. I’m usually first concerned about top level optimisation, and the code is rarely terribly complicated. When we want to debug heavy distributed training, sure, we want something more, but even with most small scale ML development this is almost all I need. 
And because I’ve got in the habit of importing my package at the top of files as a sort of default along with things like torch and numpy, I know it’s always available, and typing `gadget()` rather than print and trying to write something sensible to locate where we are in the code and grab a time stamp is so much easier. 

There’s more to the package as well, we can group calls to track the time in a certain block, reset timers, and turn everything on or off with a verbose option in the config. That’s all useful, but not essential.
 
 
For those details you can look at the README on GitHub / PiPy, the main thing this project really made me think about was friction and utility, and remembering what makes less, more. 
