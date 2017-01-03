{-# LANGUAGE TypeOperators #-}

module Volume.Types where

import Data.Array.Repa

type Weights = Array U DIM4 Double
type Volume  = Array U DIM3 Double
type Matrix  = Array U DIM2 Double
type Vector  = Array U DIM1 Double
type Bias    = Volume
type DWeights = Array D DIM4 Double
type DVolume  = Array D DIM3 Double
type DMatrix  = Array D DIM2 Double
type DVector  = Array D DIM1 Double

type Volumes = Array U DIM4 Double

type ArrayU sh = Array U sh Double
type ArrayD sh = Array D sh Double

type VectorBase s = s:.Int
type MatrixBase s = s:.Int:.Int
type TensorBase s = s:.Int:.Int:.Int
