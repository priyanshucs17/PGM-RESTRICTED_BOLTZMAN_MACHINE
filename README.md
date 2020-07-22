# PGM-RESTRICTED_BOLTZMAN_MACHINE 
## Problem Statement:
As you already know, you have to perform Topic Modelling on a set of documents and print the distribution of words in each topic.

You have seen that the training process is contrastive divergence. In this assignment, we shall create a function to perform Gibbs sampling k times within contrastive divergence often referred to as CD-k. Let's understand the structure of the notebook and the CD-k training process:

-> Importing the dataset.

-> Create a bag of words model.

-> Note that the shape of a bag of words model will be (data_size, vocablury_size)
Please note that the above shape might vary with the way you perform bag of words model but the index data_size will be the same.

-> Training using __CD-k__

-> This involves performing Gibbs sampling k-times.

1. You start with the input batch of data, v0. 

2. You then calculate p(h|v0) = σ(cj + ∑n i=1 vi.wij) as seen in the lecture. 

-> Vectorized implementation: p(h|v0)= σ(C + V.W)

3. Using this p(h|v0), you sample h0.

4. Now that you have got h0, you calculate p(v|h0)=σ(bj + ∑mj=1 hj.wij) as seen in the lecture.

-> Vectorized implementation: p(v|h0)=σ(B + H .WT)Using this p(v|h0), you sample v1.

You repeat this k times till you get hk and vk.

Step 2 and 3 are performed in the function sampleHiddenLayer() while Step 4 and 5 are performed in sampleVisibleLayer(). 

__sampleHiddenLayer() and sampleVisibleLayer()__ are combined to create a function __gibbs()__ which does one iteration of __Gibbs sampling__.

gibbs is repeated k-times in the function __cd_k()__ to perform __Contrastive Divergence k-times__.

You have already learned that the training process in an RBM involves maximizing the joint probability distribution. Using the energy function as defined in the lectures and the above sampling process, the update matrices and vectors simplify as follows:

1. ΔW = v0 ⊗ p(h0|v0) − vk ⊗ p(hk|vk)
2. Δb = avg_across_batch(vo−vk)
3. Δc = avg_across_batch(p(h0|vo) − p(hk|vk))

You do average across batch because you need a vector update for the bias vectors and vo, vk and p(h0|vo), p(hk|vk) are matrices.

Note that the exact derivations are not covered as it is very complex and the Prof. has given the intuition about the training process in the previous segments.

Since you have to maximize the joint probability distribution, you use gradient ascent here with momentum. It is recommended that you understand the working of momentum.

#### The momentum equations are as follows:
1. mWt = γm Wt−1− ΔW

2. mbt = γmbt−1 − Δb 

3. mct = γmct−1− Δc

-> γ is the momentum coefficient here.

#### Using these momentum terms, you update the weights and biases as follows:
1. Wt = Wt−1 + αmWt

2. bt = bt−1 + αmbt

3. ct = ct−1 + αmct

-> α is the learning rate.

The above equations are implemented in the function __train()__.

These equations should help you in implementing topic modelling using RBMs without any issue.


PGDML_assignment
