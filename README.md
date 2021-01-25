# Market Microstructure

Project for the MFM Workshop - Winter 2021. 

## Project Description

The project aims to explore the fundamental of market microstructure, such as order-matching algorithms, quantifying order impacts, and theoretical models such as the Hawkes process. 

## Basic

### Market Order matching algorithm

Some terminologies: 

- Market Order: order that matches at arrival
- Limit Order: order that does not match at arrival
- Limit Order Book (LOB): table recording the limit orders
- Best Bid Price: highest stated price among buy limits
- Best Ask Price: lowest stated price among sell limits
- Bid Ask Spread: Best Ask Price - Best Bid Price
- Lot Size: smallest amount of the asset that can be traded within the given LOB
- Tick Size: smallest permissible price interval between different orders

### References

- For an excellent text, see Trades, Quotes and Prices: Financial Markets
under the Microscope; Bouchard , Bonart, Donier, Gould.

- Impact of the LOB on price
  + https://arxiv.org/abs/1706.04163
- Hawkes model:
  + https://arxiv.org/abs/1412.7096
  + https://stmorse.github.io/journal/Hawkes-python.html
  + Excellent python library for Hawkes processes: https://x-datainitiative.github.io/tick/index.html

## Project

### Data Source

We looked at S&P 500 minis and WTI (Oil) futures. 

The data for this project is taken from the CME group, licensed to the UMN. The data information and code book and be found [here](https://www.cmegroup.com/confluence/display/EPICSANDBOX/CME+DataMine+Post-Purchase+Information). 

### Data Overview Preprocessing

(tbu - Ben)

### Data Visualization

(tbu)

### Impact Models

(tbu)

### Hawkes Processes

(tbu)

## Acknowledgement and Collaboration
A.M., B.S., H.P., M.N, and W.R.  
Mentor: H.D.

