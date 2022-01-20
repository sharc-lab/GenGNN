*Optimizations:*

- Fully parallelized the MLP\_1\_INPUT dimension and MLP\_2\_OUTPUT dimensions.

- Saves all the five-layer weights on-board.

- Successfully parallelized the MLP with message passing. Improved the latency from 0.91ms to 0.82ms to 0.55ms (verified on U280). A little slowdown on U50.

*Version*

- Fixed pipeline of message passing and node embedding
