{-# OPTIONS_GHC -Odph -rtsopts -threaded -fno-liberate-case
                -funfolding-use-threshold1000 -funfolding-keeness-factor1000
                -fllvm -optlo-O3 #-}

{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE Strict #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE DeriveGeneric #-}

module Volume where

import Label
import Data.Word
import Data.Array.Repa as R hiding ((++))
import qualified Data.Vector.Unboxed as DV
import Data.Array.Repa.Algorithms.Randomish
import GHC.Generics (Generic)
import Data.Serialize
import Volume.Internal
import Volume.Types
import Control.Monad


-- | A network layer that has volumes as both its input and output.
data Layer3
  = Conv -- ^ A layer that applies a convolution of its weights (or,
         --   more accurately, a correlation) to its input volume.
         Weights -- ^ The weights used for the convolution. The first
                 --   dimension, i, is used to distinguish different weights.
                 --   Its second dimension, d, is equal to the input volume's
                 --   first dimension d, and therefore disappears jrom the output
                 --   in order to produce a volume again.
         Bias    -- ^ A bias volume that is added to the output volume.
  | ReLU -- ^ A rectified linear unit, or ReLU. NN jargon for (\x -> max x 0).
         --   A ReLU is cheap but has all the properties that you want from an
         --   activation function and hence is the most commonly used.
  | Pool -- ^ A max-pooling layer. The pool size has been hard-coded to 2, at least
         --   for now. A pooling layer subsamples the input to a quarter the size,
         --   passing through the maximum element in each 2x2 subregion.
         deriving (Eq, Generic)
instance Serialize Layer3

instance Show Layer3 where
  show Pool       = "Pool"
  show ReLU       = "ReLU"
  show (Conv w b) = "Conv dimW: " ++ dimW ++ " dimB: " ++ dimB
    where
      dimW = show . extent $ w
      dimB = show . extent $ b

-- | Apply a volume to a Layer3
forward :: (Monad m, Shape sh)
        => ArrayU (sh:.Int:.Int:.Int)
        -> Layer3
        -> m (ArrayU (sh:.Int:.Int:.Int))
forward x (Conv w b) = w `corr` x >>= computeP . (addConform b)

forward x ReLU       = computeP $ R.map (max 0) x
forward x Pool       = pool x

softMax :: (Shape sh, Monad m) => ArrayU (VectorBase sh) -> [Int] -> m (ArrayU (VectorBase sh))
softMax x cs = do let b:.i = extent x
                      offsets = scanl (+) 0 cs
                      ranges = zip offsets cs
                      extractRange (ix,w) = extract (b:.ix) (unitDim:.w) x
                      cols = extractRange <$> ranges
                      expNorm   ps   maxs = computeP $ traverse2 ps   maxs const (\f1 f2 ix@(b:._) -> exp$ f1 ix - f2 b)
                      subtract' exps sums = traverse2 exps sums const (\f1 f2 ix@(b:._) -> f1 ix / f2 b)

                  maxElems <- Prelude.traverse (foldP max (-1/0)) cols
                  exps :: [ArrayU (sh:.Int)]  <- zipWithM expNorm cols maxElems
                  expSums :: [ArrayU sh]      <- Prelude.traverse sumP exps
                  let probs :: [ArrayD (sh:.Int)] = Prelude.zipWith subtract' exps expSums
                  computeP$ foldl1 append probs

splitCs :: [Int] -> DV.Vector Double -> [DV.Vector Double]
splitCs [] _ = []
splitCs (c:cs) rem = let (h,t) = DV.splitAt c rem in h:splitCs cs t

normalize :: DV.Vector Double -> DV.Vector Double
normalize vec = DV.map (/s) vec
  where s = DV.sum vec

softMaxBackward :: Monad m => Vector -> [Int] -> [Label] -> m (Vector, [Double])
softMaxBackward y cs ls = do dx <- computeP$ R.traverse y id lkFn
                             return (dx, losses)
  where
    offsets = scanl (+) 0 cs
    ixs     = Prelude.zipWith (+) offsets (fromLabel <$> ls)
    losses  = dataLoss y . toLabel <$> ixs
    lkFn lkUp (Z:.i) = if i `elem` ixs then lkUp (ix1 i) -1 else lkUp (ix1 i)

getMaxima :: Vector -> [Int] -> [Int]
getMaxima vec cs = DV.maxIndex <$> splitCs cs (toUnboxed vec)

findThreshold :: Double -> Vector -> Label
findThreshold t = maybe Indeterminate toLabel . DV.findIndex (>t) . toUnboxed

getMaximaThresholded ::  Vector -> [Int] -> Double -> [Label]
getMaximaThresholded vec cs t = find <$> vecs
  where
    vecs = splitCs cs (toUnboxed vec)
    find vec = maybe Indeterminate toLabel . DV.findIndex (>t) $ vec

-- | Propagate an error gradient backwards through a Layer3. Some arguments
--   are calculated during the forward pass. We could recalculate them
--   during the backwards pass, but for the sake of both efficiency
--   and clarity I chose to reuse them from the forward pass. The recursive
--   definition of the training function makes this work out quite nicely.
backward :: (Shape sh, Monad m)
         => Layer3  -- ^ Layer to backprop through
         -> ArrayU (TensorBase sh)  -- ^ Input for this layer during the forward pass
         -> ArrayU (TensorBase sh)  -- ^ Output for this layer during the forward pass
         -> ArrayU (TensorBase sh)  -- ^ Error gradient on the output of this layer
         -> m (Layer3, ArrayU (TensorBase sh)) -- ^ Weight deltas, and error gradient on this layer's input.
backward (Conv w _) x _ dy =
  do dx <- w `fullConv` dy
     dw <- dy `corrVolumes` x
     db <- computeP$ sumOuter dy
     return (Conv dw db, dx)

backward Pool x y dy =
  do dx <- poolBackprop x y dy
     return (Pool, dx)

backward ReLU _ y dy =
  do dx <- computeP $ R.zipWith (\x t -> if t > 0 then x else 0) dy y
     return (ReLU, dx)

applyDelta :: Monad m
           => Layer3
           -> Layer3
           -> Layer3
           -> Double
           -> Double
           -> Double
           -> m (Layer3, Layer3)
applyDelta (Conv dw db) (Conv w b) (Conv vw vb) α λ γ =
  do vw' <- computeP$ R.zipWith (\v d -> γ*v - α*d) vw dw
     vb' <- computeP$ R.zipWith (\v d -> γ*v - α*d) vb db
     w'  <- computeP$ R.zipWith (\w v -> w + v - λ*w) w vw'
     b'  <- computeP$ R.zipWith (\b v -> b + v - λ*b) b vb'
     return (Conv w' b', Conv vw' vb')

applyDelta _ l v _ _ _ = return (l,v)

initVelocity :: Layer3 -> Layer3
initVelocity (Conv w b) = Conv (computeS$ R.map (const 0) w) (computeS$ R.map (const 0) b)
initVelocity x = x

-- | Rotates the two topmost dimensions of an array by 180 degrees.
{-# INLINE rotate #-}
rotate :: (Source r e, Shape tail) => Array r ((tail :. Int) :. Int) e -> Array D ((tail :. Int) :. Int) e
rotate arr = backpermute sh invert arr
  where
    sh@(_:.h:.w) = extent arr
    invert (b:.y:.x) = b:.h-y-1:.w-x-1

-- | Rotates the two topmost dimensions of an array by 180 degrees.
{-# INLINE rotateW #-}
rotateW :: Weights -> DWeights
rotateW arr = backpermute (Z:.d:.n:.h:.w) invert arr
  where
    Z:.n:.d:.h:.w = extent arr
    invert (Z:.i:.z:.y:.x) = Z:.z:.i:.h-y-1:.w-x-1


conv :: Monad m => Volume -> Volume -> m Weights
conv krn img = do krn' <- computeP $ rotate krn
                  corrVolumes krn' img

fullConv :: (Shape sh, Monad m) => Weights -> ArrayU (TensorBase sh) -> m (ArrayU (TensorBase sh))
fullConv krn img = do krn' <- computeP $ rotateW krn
                      img' <- computeP $ zeropad (kh-1) img
                      krn' `corr` img'
  where
    _:.kh:._ = extent krn

{-# INLINE zeropad #-}
zeropad :: (Source r Double, Shape tail) => Int -> Array r ((tail :. Int) :. Int) Double -> Array D ((tail :. Int) :. Int) Double
zeropad n a = R.traverse a shFn padFn
  where
    _:.h:.w = extent a

    {-# INLINE shFn #-}
    shFn (b:.h:.w) = b:.h+2*n:.w+2*n
    {-# INLINE padFn #-}
    padFn lkFn (b:.y:.x)
      | y < n || y >= h + n || x < n || x >= w + n = 0
      | otherwise = lkFn (b:.y-n:.x-n)

{-# INLINE vmmult #-}
vmmult :: (Source r Double, Source r2 Double) => Array r2 DIM1 Double -> Array r DIM2 Double -> Array D DIM1 Double
vmmult v m = fromFunction sh' ixFn
  where
    Z:._:.c = extent m
    sh' = Z:.c
    ixFn (Z:.i :: DIM1) = sumAllS $ slice m (Any:.i) *^ v

-- | Vector-vector multiplication. The output is a matrix in which every
--   (x,y) position corresponds to the xth element of the first vector
--   and the yth element of the second. Hence, the resulting matrix is
--   as wide as the first vector's length and as tall as the second's.
{-# INLINE vvmult #-}
vvmult :: (Source r2 Double, Source r1 Double) => Array r1 DIM1 Double -> Array r2 DIM1 Double -> Array D DIM2 Double
vvmult vw vh = traverse2 vw vh shFn vFn
  where
    shFn (Z:.w) (Z:.h) = Z:.h:.w
    vFn v1 v2 (Z:.y:.x) = v1 (ix1 x) * v2 (ix1 y)

-- | Data loss of a network output, given some classification.
--   This value is somewhere between 0 (p_correct == 1) and
--   infinity (p_correct == 0).
dataLoss :: Vector -> Label -> Double
dataLoss p (fromLabel -> i) = negate . log $ linearIndex p i

-- | The label of some input to a network, as determined by
--   the output of the network for that input.
--   If the output layer of the network is a SoftMax function,
--   the class scores produced by that network can be interpreted
--   as probability/certainty scores. This function returns the
--   class the network assigns the highest probability to and
--   converts it into a label.
maxIndex :: Vector -> Label
maxIndex = toLabel . DV.maxIndex . toUnboxed

maxElem :: (Source r Double, Shape sh, Monad m) => Array r sh Double -> m Double
maxElem = foldAllP max (-1/0)

minElem :: (Source r Double, Shape sh, Monad m) => Array r sh Double -> m Double
minElem = foldAllP min (1/0)

subtractOneAt :: Monad m => Int -> Vector -> m Vector
subtractOneAt i arr = computeP $ R.traverse arr id ixFn
  where ixFn src (Z:.w) = if w == i then src (ix1 w) - 1 else src (ix1 w)

-- TODO: Non-urgent; Let forward1 accept delayed representations so the
-- flattened array does not need to be rebuilt in memory
flatten :: (Monad m, Source r1 Double, Shape sh1) => Array r1 sh1 Double -> m Vector
flatten arr = computeP $ reshape (Z:.s) arr
  where s = product . listOfShape . extent $ arr

randomConvLayer :: Int -- ^ Kernel size
                -> Int -- ^ Kernel/input depth
                -> Int -- ^ Kernel count
                -> Int -- ^ Input width
                -> Int -- ^ Input height
                -> Int -- ^ RNG Seed
                -> Layer3
randomConvLayer ks kd kn iw ih seed = Conv w b
  where
    w = randomishDoubleArray (Z:.kn:.kd:.ks:.ks)       1e-2 (-1e-2) seed
    b = randomishDoubleArray (Z:.kn:.ih-ks+1:.iw-ks+1) 1e-2 (-1e-2) (seed+1)

toCifarVolume :: [Word8] -> Volume
toCifarVolume = fromListUnboxed (Z:.3:.32:.32) . fmap ((/255) . fromIntegral)

lerp :: (Source r Double, Shape sh, Monad m) => Array r sh Double -> Double -> Double -> m (Array U sh Double)
lerp arr lo hi = do minE <- minElem arr
                    maxE <- maxElem arr
                    computeP $ R.map (\x -> lo + (x-minE) * (hi-lo) / (maxE-minE)) arr

splitW :: Monad m => Weights -> m [Volume]
splitW arr = Prelude.traverse computeP slices
  where
    Z:.n:._:._:._ = extent arr
    slices :: [DVolume]
    slices = [ slice arr (Z:.ni:.All:.All:.All) | ni <- [0..n-1]]

cropFast :: Monad m => Int -> Int -> Int -> Int -> Int -> Array U DIM2 (Word8, Word8, Word8) -> m Volume
cropFast r rx ry rw rh img = computeP$ fromFunction (Z:.3:.r:.r) lookup
  where
    Z:.h:._ = extent img
    {-# INLINE toX #-}
    toX x = rx + (x * rw) `div` r
    {-# INLINE toY #-}
    toY y = h - ry - (y * rh) `div` r
    {-# INLINE lookup #-}
    lookup (Z:.0:.y:.x) = let (r,_,_) = img `unsafeIndex` (ix2 (toY y) (toX x)) in fromIntegral r / 255
    lookup (Z:.1:.y:.x) = let (_,g,_) = img `unsafeIndex` (ix2 (toY y) (toX x)) in fromIntegral g / 255
    lookup (Z:.2:.y:.x) = let (_,_,b) = img `unsafeIndex` (ix2 (toY y) (toX x)) in fromIntegral b / 255
    lookup _ = undefined

volToBmp :: Monad m => Volume -> m (Array U DIM2 (Word8, Word8, Word8))
volToBmp vol = computeP $ R.traverse vol (\(_:.h:.w) -> Z:.h:.w) fn
  where
    fn f (Z:.y:.x) = (f' 0 y x, f' 1 y x, f' 2 y x)
      where f' n y x = round . (*255) $ f (ix3 n y x)

getWeights :: [Layer3] -> [Weights]
getWeights [] = []
getWeights (Conv w _:ws) = w : getWeights ws
getWeights (_:ws) = getWeights ws

instance (DV.Unbox e, Serialize e, Shape sh) => Serialize (Array U sh e) where
  put a = do put (listOfShape . extent $ a)
             put (toList a)
  get   = do sh    <- shapeOfList <$> get
             elems <- get
             return (fromListUnboxed sh elems)
