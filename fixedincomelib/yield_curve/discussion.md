
Model :

    - vdate : as of which date you're building model
    - model_type : e.g., YIELD_CURVE, IR_SABR, IR_LGM ...
    - data_collection : a collection of data objects to calibrate your model
    - build_method_collection : a collection of build methods

Data Colleciton : 

    - collection of data objects
    - each data object is identified by (data_type, data convention)
        - for instance, suppose we want to construct a SOFR-1B component (of a Yield Curve)
            - RFR Futures
            - RFR Swaps

            Ideally, we want to create data objects for each of them. Take Future as an example, we want an API
            createData1D which takes
                - data_type : RFR Future
                - data_conv : SOFR-FUTURE-3M
                - axis1 : 2025-09-20 x 2025-12-20, .....
                - values : 100.0, 99.8, ....
            Similarly, for swap, we want 
                - data_type : RFR Swap
                - data_conv : USD-SOFR-OIS
                - axis1 : 4Y, 5Y, ...
                - values : 3%, 4%

Build Method Collection:

    - collection of build methods
    - each build method is identified by (target, build_method_type)
        - for instance, we are constructing a yield curve with multiple components (SOFR-1B, USD-LIBOR-BBA-3M, ...)
        - so the target is SOFR-1B, USD-LIBOR-BBA-3M ...
        - For each target, say, SOFR-1B, we need to describe the recipe of building this component
        - How ?

            TARGET              : SOFR-1B
            REFERENCE           :
            FUTURE              : SOFR-FUTURE-3M
            SWAP                : USD-SOFR-OIS
            FRA                 : 
            BASIS SWAP          : 
            INSTANTANEOUS FR    : 
            INTERPLATION_METHOD : PIECEWISE_CONSTANT_LEFT_CONTINUOUS
            
        - Another component, can be constructed differently, 

            TARGET              : USD-LIBOR-BBA-3M
            REFERENCE           :
            FUTURE              : EURODOLLAR
            SWAP                : USD-SWAP-SEMI-BOND
            FRA                 : USD-BBA-3M-FRA
            BASIS SWAP          : 
            INSTANTANEOUS FR    : 
            INTERPLATION_METHOD : PIECEWISE_CONSTANT_LEFT_CONTINUOUS

            TARGET              : USD-LIBOR-BBA-6M
            REFERENCE           : USD-LIBOR-BBA-3M
            FUTURE              : 
            SWAP                : 
            FRA                 : 
            BASIS SWAP          : USD-LIBOR-BBA-6M-OVER-3M
            INSTANTANEOUS FR    : 
            INTERPLATION_METHOD : PIECEWISE_CONSTANT_LEFT_CONTINUOUS


So then how it works ?

The yield curve construction boils down to the following problem:

- We loop through build method 
    - this is tricky, you need to figure out figure the order of construction of components
      because sometimes Component A depends on component B, C, then you have to construct B and C 
      first before constructing A

- Suppose we are constructing component X, then we go through the recipe

    - check each NVP, e.g., Swap => do we have anything ? 
        - If yes, we go to data collection, fish out (Swap, convention you find in the build method) => fish out the data objects
        - if not, I pass
    
    - After a loop, I collected all instruments that i needed to create this component
    - Now the question becomes, how do I calibrate model parameters to these instruments

    Model is parameterized by X^I, Model receive the data collection
    Each componet have a subset of X^I(T), for which we already collected the relevant instrments X^M(T), 
    Remember you have your createProductFromDataConvention, that means, you give it axis1, data convention, data type, 
    values, it create the productcg P(X^M(T)_i). For instance, 

     data objects (data_type, RFR Future)
    - axis1 : 2025-09-20 x 2025-12-20, .....
    - values : 100.0, 99.8, ....
    cretaeProductFromDataConvention(data_type, convention, axis1[i], values[i])
    

    - First sort X^M(T) by its last date of the product, and this initilaize you X^I(T) = 0
    - Now, you start bootstrapping or global solve. Let us talk about bootstrapping:

        - M, P(X^M(T)_i) => ValutionEngineRegistry => ValuationEngine(M, P) = 0
        - then you continue, the moment you finish, your X^I(T) is all finalized


Now, where to start, let's do the SIMPLEST calibration (because I want you to do it without valuationEngine).
So what can we do ? We direcly let X^I = X^M, that is your instrument is internal parameter, or in other words,
your X^M is the instantaneous forward rate. 

Data Type : Instantaneous Forward Rate
Data Conv : SOFR-1B-IFR
Axis1 : 1M, 1Y, 2Y, .....
Values : 0.03, 0.05, ......

Then your calibration is trivial, you will have the below build method

TARGET              : SOFR-1B
REFERENCE           : 
FUTURE              : 
SWAP                : 
FRA                 : 
BASIS SWAP          : 
INSTANTANEOUS FR    : SOFR-1B-IFR
INTERPLATION_METHOD : PIECEWISE_CONSTANT_LEFT_CONTINUOUS

So you do the same, you loop through buld method name-value pairs, you found the only thing exists there
is IFR, then your calibratino is trivial 

- you still sort your X^M, but that's nothing, it's just the anchored poitn 1M, 1Y, .....
- you calibrate, but that's nothing, it' just X^I = X^M for all these tenors

Then you're done.


