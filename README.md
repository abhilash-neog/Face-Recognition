------------------------------------------------------------------------------------------------------
Hybrid GSA-SDS-BBBC Algorithm using Chaotic Local Search & It's Application to a Face Recognition Scenario
------------------------------------------------------------------------------------------------------

## Setup

**1)Language:** Python 3.6(recommended)

**2)Libraries required:** numpy, matplotlib, sklearn, scipy

**OR**

**3) Anaconda 4 :** [Download](https://anaconda.org/)


## Functions Description: 
          
**(i) CostFun->**  The ojective function used to evaluate data   points
	inputs : Face Vector Matrix, Data point
	outputs: objective function value
                
**(ii) Evaluate->**  Evaluates the objectve function for all the eigen vectors and return fitness matrix
	inputs : Eigen Matrix, Face Vector Matrix
	outputs: fitness matrix
	        
**(iii) MassCalc->**  Updates the masses of our eigen vectors based on their fitness values
	inputs : fitness values matrix
	outputs: Updated Mass Array of eigen vectors M
	   
**(iv)  Gconstant->**  Gives the G value for current iteration
	inputs : current iteration, max iterations
	outputs: G value
	         
**(v) Gfield ->**  Updates the acceleration values
	inputs : G, M, S, iteration number, Max iterations
	outputs: acceleration matrix a
	        
**(vi)  move->**  Gives the updated position and velocity matrices
	inputs : position matrix S, acceleration matrix a, velocity matrix V
	outputs: S,V

**(vii) Sbound ->**  Checks if all the points are within the specified boundaries. 
                If not, re-initializes them randomly within the boundaries
	inputs : S, upper limit, lower limit
	outputs: S

**(viii)ChaoticLocalSearch->**  Thorough searching for better solutions around a best solution
	inputs : Face matrix, center point
	outputs: best eigen vector, it's ojective funtion
	       
**(ix)  NewChaos->**  Initializes a new chaos variable
	inputs : None
	outputs: random value between 0 and 1

**(x) plot_gallery->**  plots the eigen faces obtained
	inputs : eigen face vectors, titles, height, width
	outputs: eigen vectors figure

**(xi) checkAgents->** checks whether an agent is active or passive and accordingly classifies them
	inputs : dislpacement of agents(an array), current threshold 
	outputs : passive and active agents
	   
**(xii) diffusion->** Causes the diffusion of hypothesis by the active diffusion process
	inputs : active agents, passive agents, fitness array
	outputs : updated fitness array

**(xiii) thresholdValue ->** updates the threshold after every iteration
	inputs: current iteration, current threshold value
	output: new threshold value

**(xiv) calcDisplacement->** calculates the displacement of every agent in each iteration
	inputs: current search space, old search space(i.e. positions of agents)
	output: difference in position of agents(i.e. displacement) 

## Working 

The code automatically fetches the dataset by connecting to the internet. Number of components is given as 150 (can be changed as per the programmer's wish).Prinicple Component Analysis is done on the dataset and eigen vectors are extracted. Then dimensionality reduction is done on the eigen vectors to match the number of components. This eigen vector matrix is given as the input to the algorithm with the objective function being the sum of it's corresponding weights in each face vector. The dataset would be represented by the transformed eigen vectors. Then it's divided into train and test data and passed on to an SVM classifier. The classification is done and results are noted.

