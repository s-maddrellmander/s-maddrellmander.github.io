---
title: "Alignment through the lens of a darts playing bot"
description: "Reflecting on alignment difficulties in a simple darts playing RL environment."
categories: [Projects]
tags: [RL]
date: 2025-12-15 17:30:00 +0000

---
# That was hard.

That was hard. It shouldn't have been that difficult. But it was. 
I'm not an RL expert, that's for sure, but come on? This is such a simple game. 

For anyone who hasn't been in as many english pubs let me give a couple of words on the game of darts. 
Ignoring all the usual criticisms about people who lack conversation.

The board is a circle divided into 20 equal segments, and on each segment there are two thin bands which score doubel or triple the points. 
There's also a bullseye inner and outter with 25 and 50 points resptively. 
And the aim of the game is to get from your starting score to 0, faster than your oppoennt, where you each get three darts at a time. 
Oh and to actually win you must end on a double (or bullseye) .
That's basically it. 

It feels like such a trivially simple case to make a reinforcemnt learning model work - all it needs it to aim at the board, and over time work out that 
some areas are worth more than others, then which parts to aim to win. 
Given that there are only 21 valid final single throws, and the rest don't matter so long as you score enough to get within 40 of the finish. 
It's easily solvable with a determinsitic algorithm. 

But - due to my incompotence or a few subtle difficulties - that wasn't the case. This is that story. 


# What's the setup? 

