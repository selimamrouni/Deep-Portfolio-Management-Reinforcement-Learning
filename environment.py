import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from gym.envs.registration import register

class TradeEnv():

    """
    This class is the trading environment (render) of our project. 

    The trading agent calls the class by giving an action at the time t. 
    Then the render gives back the new portfolio at the next step (time t+1). 

    #parameters:

    - windonw_length: this is the number of time slots looked in the past to build the input tensor
    - portfolio_value: this is the initial value of the portfolio 
    - trading_cost: this is the cost (in % of the traded stocks) the agent will pay to execute the action 
    - interest_rate: this is the rate of interest (in % of the money the agent has) the agent will:
        -get at each step if he has a positive amount of money 
        -pay if he has a negative amount of money
    -train_size: % of data taken for the training of the agent - please note the training data are taken with respect 
    of the time span (train -> | time T | -> test)
    """

    def __init__(self, path = './np_data/input.npy', window_length=50,
                 portfolio_value= 10000, trading_cost= 0.25/100,interest_rate= 0.02/250, train_size = 0.7):
        
        #path to numpy data
        self.path = path
        #load the whole data
        self.data = np.load(self.path)


        #parameters
        self.portfolio_value = portfolio_value
        self.window_length=window_length
        self.trading_cost = trading_cost
        self.interest_rate = interest_rate

        #number of stocks and features
        self.nb_stocks = self.data.shape[1]
        self.nb_features = self.data.shape[0]
        self.end_train = int((self.data.shape[2]-self.window_length)*train_size)
        
        #init state and index
        self.index = None
        self.state = None
        self.done = False

        #init seed
        self.seed()

    def return_pf(self):
        """
        return the value of the portfolio
        """
        return self.portfolio_value
        
    def readTensor(self,X,t):
        ## this is not the tensor of equation 18 
        ## need to batch normalize if you want this one 
        return X[ : , :, t-self.window_length:t ]
    
    def readUpdate(self, t):
        #return the return of each stock for the day t 
        return np.array([1+self.interest_rate]+self.data[-1,:,t].tolist())

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self, w_init, p_init, t=0 ):
        
        """ 
        This function restarts the environment with given initial weights and given value of portfolio

        """
        self.state= (self.readTensor(self.data, self.window_length) , w_init , p_init )
        self.index = self.window_length + t
        self.done = False
        
        return self.state, self.done

    def step(self, action):
        """
        This function is the main part of the render. 
        At each step t, the trading agent gives as input the action he wants to do. So, he gives the new value of the weights of the portfolio. 

        The function computes the new value of the portfolio at the step (t+1), it returns also the reward associated with the action the agent took. 
        The reward is defined as the evolution of the the value of the portfolio in %. 

        """

        index = self.index
        #get Xt from data:
        data = self.readTensor(self.data, index)
        done = self.done
        
        #beginning of the day 
        state = self.state
        w_previous = state[1]
        pf_previous = state[2]
        
        #the update vector is the vector of the opening price of the day divided by the opening price of the previous day
        update_vector = self.readUpdate(index)

        #allocation choice 
        w_alloc = action
        pf_alloc = pf_previous
        
        #Compute transaction cost
        cost = pf_alloc * np.linalg.norm((w_alloc-w_previous),ord = 1)* self.trading_cost
        
        #convert weight vector into value vector 
        v_alloc = pf_alloc*w_alloc
        
        #pay transaction costs
        pf_trans = pf_alloc - cost
        v_trans = v_alloc - np.array([cost]+ [0]*self.nb_stocks)
        
        #####market prices evolution 
        #we go to the end of the day 
        
        #compute new value vector 
        v_evol = v_trans*update_vector

        
        #compute new portfolio value
        pf_evol = np.sum(v_evol)
        
        #compute weight vector 
        w_evol = v_evol/pf_evol
        
        
        #compute instanteanous reward
        reward = (pf_evol-pf_previous)/pf_previous
        
        #update index
        index = index+1
        
        #compute state
        
        state = (self.readTensor(self.data, index), w_evol, pf_evol)
        
        if index >= self.end_train:
            done = True
        
        self.state = state
        self.index = index
        self.done = done
        
        return state, reward, done
        
        
        
        
        
        
 