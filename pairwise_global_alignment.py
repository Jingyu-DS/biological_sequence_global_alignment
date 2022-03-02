import numpy as np
# read in BLOSUM50 matrix to use
blosum50 = {'A': {'A': 5.0, 'C': -1.0, 'D': -2.0, 'E': -1.0, 'F': -3.0, 'G': 0.0, 'H': -2.0, 'I': -1.0, 'K': -1.0, 'L': -2.0, 'M': -1.0, 'N': -1.0, 'P': -1.0, 'Q': -1.0, 'R': -2.0, 'S': 1.0, 'T': 0.0, 'V': 0.0, 'W': -3.0, 'Y': -2.0},
'C': {'A': -1.0, 'C': 13.0, 'D': -4.0, 'E': -3.0, 'F': -2.0, 'G': -3.0, 'H': -3.0, 'I': -2.0, 'K': -3.0, 'L': -2.0, 'M': -2.0, 'N': -2.0, 'P': -4.0, 'Q': -3.0, 'R': -4.0, 'S': -1.0, 'T': -1.0, 'V': -1.0, 'W': -5.0, 'Y': -3.0},
'D': {'A': -2.0, 'C': -4.0, 'D': 8.0, 'E': 2.0, 'F': -5.0, 'G': -1.0, 'H': -1.0, 'I': -4.0, 'K': -1.0, 'L': -4.0, 'M': -4.0, 'N': 2.0, 'P': -1.0, 'Q': 0.0, 'R': -2.0, 'S': 0.0, 'T': -1.0, 'V': -4.0, 'W': -5.0, 'Y': -3.0},
'E': {'A': -1.0, 'C': -3.0, 'D': 2.0, 'E': 6.0, 'F': -3.0, 'G': -3.0, 'H': 0.0, 'I': -4.0, 'K': 1.0, 'L': -3.0, 'M': -2.0, 'N': 0.0, 'P': -1.0, 'Q': 2.0, 'R': 0.0, 'S': -1.0, 'T': -1.0, 'V': -3.0, 'W': -3.0, 'Y': -2.0},
'F': {'A': -3.0, 'C': -2.0, 'D': -5.0, 'E': -3.0, 'F': 8.0, 'G': -4.0, 'H': -1.0, 'I': 0.0, 'K': -4.0, 'L': 1.0, 'M': 0.0, 'N': -4.0, 'P': -4.0, 'Q': -4.0, 'R': -3.0, 'S': -3.0, 'T': -2.0, 'V': -1.0, 'W': 1.0, 'Y': 4.0},
'G': {'A': 0.0, 'C': -3.0, 'D': -1.0, 'E': -3.0, 'F': -4.0, 'G': 8.0, 'H': -2.0, 'I': -4.0, 'K': -2.0, 'L': -4.0, 'M': -3.0, 'N': 0.0, 'P': -2.0, 'Q': -2.0, 'R': -3.0, 'S': 0.0, 'T': -2.0, 'V': -4.0, 'W': -3.0, 'Y': -3.0},
'H': {'A': -2.0, 'C': -3.0, 'D': -1.0, 'E': 0.0, 'F': -1.0, 'G': -2.0, 'H': 10.0, 'I': -4.0, 'K': 0.0, 'L': -3.0, 'M': -1.0, 'N': 1.0, 'P': -2.0, 'Q': 1.0, 'R': 0.0, 'S': -1.0, 'T': -2.0, 'V': -4.0, 'W': -3.0, 'Y': 2.0},
'I': {'A': -1.0, 'C': -2.0, 'D': -4.0, 'E': -4.0, 'F': 0.0, 'G': -4.0, 'H': -4.0, 'I': 5.0, 'K': -3.0, 'L': 2.0, 'M': 2.0, 'N': -3.0, 'P': -3.0, 'Q': -3.0, 'R': -4.0, 'S': -3.0, 'T': -1.0, 'V': 4.0, 'W': -3.0, 'Y': -1.0},
'K': {'A': -1.0, 'C': -3.0, 'D': -1.0, 'E': 1.0, 'F': -4.0, 'G': -2.0, 'H': 0.0, 'I': -3.0, 'K': 6.0, 'L': -3.0, 'M': -2.0, 'N': 0.0, 'P': -1.0, 'Q': 2.0, 'R': 3.0, 'S': 0.0, 'T': -1.0, 'V': -3.0, 'W': -3.0, 'Y': -2.0},
'L': {'A': -2.0, 'C': -2.0, 'D': -4.0, 'E': -3.0, 'F': 1.0, 'G': -4.0, 'H': -3.0, 'I': 2.0, 'K': -3.0, 'L': 5.0, 'M': 3.0, 'N': -4.0, 'P': -4.0, 'Q': -2.0, 'R': -3.0, 'S': -3.0, 'T': -1.0, 'V': 1.0, 'W': -2.0, 'Y': -1.0},
'M': {'A': -1.0, 'C': -2.0, 'D': -4.0, 'E': -2.0, 'F': 0.0, 'G': -3.0, 'H': -1.0, 'I': 2.0, 'K': -2.0, 'L': 3.0, 'M': 7.0, 'N': -2.0, 'P': -3.0, 'Q': 0.0, 'R': -2.0, 'S': -2.0, 'T': -1.0, 'V': 1.0, 'W': -1.0, 'Y': 0.0},
'N': {'A': -1.0, 'C': -2.0, 'D': 2.0, 'E': 0.0, 'F': -4.0, 'G': 0.0, 'H': 1.0, 'I': -3.0, 'K': 0.0, 'L': -4.0, 'M': -2.0, 'N': 7.0, 'P': -2.0, 'Q': 0.0, 'R': -1.0, 'S': 1.0, 'T': 0.0, 'V': -3.0, 'W': -4.0, 'Y': -2.0},
'P': {'A': -1.0, 'C': -4.0, 'D': -1.0, 'E': -1.0, 'F': -4.0, 'G': -2.0, 'H': -2.0, 'I': -3.0, 'K': -1.0, 'L': -4.0, 'M': -3.0, 'N': -2.0, 'P': 10.0, 'Q': -1.0, 'R': -3.0, 'S': -1.0, 'T': -1.0, 'V': -3.0, 'W': -4.0, 'Y': -3.0},
'Q': {'A': -1.0, 'C': -3.0, 'D': 0.0, 'E': 2.0, 'F': -4.0, 'G': -2.0, 'H': 1.0, 'I': -3.0, 'K': 2.0, 'L': -2.0, 'M': 0.0, 'N': 0.0, 'P': -1.0, 'Q': 7.0, 'R': 1.0, 'S': 0.0, 'T': -1.0, 'V': -3.0, 'W': -1.0, 'Y': -1.0},
'R': {'A': -2.0, 'C': -4.0, 'D': -2.0, 'E': 0.0, 'F': -3.0, 'G': -3.0, 'H': 0.0, 'I': -4.0, 'K': 3.0, 'L': -3.0, 'M': -2.0, 'N': -1.0, 'P': -3.0, 'Q': 1.0, 'R': 7.0, 'S': -1.0, 'T': -1.0, 'V': -3.0, 'W': -3.0, 'Y': -1.0},
'S': {'A': 1.0, 'C': -1.0, 'D': 0.0, 'E': -1.0, 'F': -3.0, 'G': 0.0, 'H': -1.0, 'I': -3.0, 'K': 0.0, 'L': -3.0, 'M': -2.0, 'N': 1.0, 'P': -1.0, 'Q': 0.0, 'R': -1.0, 'S': 5.0, 'T': 2.0, 'V': -2.0, 'W': -4.0, 'Y': -2.0},
'T': {'A': 0.0, 'C': -1.0, 'D': -1.0, 'E': -1.0, 'F': -2.0, 'G': -2.0, 'H': -2.0, 'I': -1.0, 'K': -1.0, 'L': -1.0, 'M': -1.0, 'N': 0.0, 'P': -1.0, 'Q': -1.0, 'R': -1.0, 'S': 2.0, 'T': 5.0, 'V': 0.0, 'W': -3.0, 'Y': -2.0},
'V': {'A': 0.0, 'C': -1.0, 'D': -4.0, 'E': -3.0, 'F': -1.0, 'G': -4.0, 'H': -4.0, 'I': 4.0, 'K': -3.0, 'L': 1.0, 'M': 1.0, 'N': -3.0, 'P': -3.0, 'Q': -3.0, 'R': -3.0, 'S': -2.0, 'T': 0.0, 'V': 5.0, 'W': -3.0, 'Y': -1.0},
'W': {'A': -3.0, 'C': -5.0, 'D': -5.0, 'E': -3.0, 'F': 1.0, 'G': -3.0, 'H': -3.0, 'I': -3.0, 'K': -3.0, 'L': -2.0, 'M': -1.0, 'N': -4.0, 'P': -4.0, 'Q': -1.0, 'R': -3.0, 'S': -4.0, 'T': -3.0, 'V': -3.0, 'W': 15.0, 'Y': 2.0},
'Y': {'A': -2.0, 'C': -3.0, 'D': -3.0, 'E': -2.0, 'F': 4.0, 'G': -3.0, 'H': 2.0, 'I': -1.0, 'K': -2.0, 'L': -1.0, 'M': 0.0, 'N': -2.0, 'P': -3.0, 'Q': -1.0, 'R': -1.0, 'S': -2.0, 'T': -2.0, 'V': -1.0, 'W': 2.0, 'Y': 8.0}}

# X and Y are two sequences; track_direction is a list to record all the steps taken to get the global maximum
def DP_GLOBAL_ALIGHMENT(X, Y):
  n = len(X)
  m = len(Y)
  d = 8 # penalty for gap 

  F = np.zeros((n+1, m+1))
  T = np.zeros((n+1, m+1))

  for i in range(0, n+1):
    F[i, 0] = i * (-d)
    T[i, 0] = 2 # Use 2 to denote left direction
  
  for j in range(1, m+1):
    F[0, j] = j * (-d)
    T[0, j] = 3 # Use 3 to denote up direction
  
  for i in range(1, n+1):
    for j in range(1, m+1):
      Option1 = F[i-1, j-1] + blosum50[X[i-1]][Y[j-1]] # diagonal
      Option2 = F[i-1, j] - d # left
      Option3 = F[i, j-1] - d # up
      F[i, j] = max(Option1, Option2, Option3)
      num_direction = {1: Option1, 2: Option2, 3: Option3}
      T[i, j] = max(num_direction, key=num_direction.get)
  
 
  # 1: append the letter from both sequences 
  # 2: append gap in y
  # 3: append gap in x
  track_direction = []
  track_direction = track_back(T, n, m, track_direction)

  X_match = []
  Y_match = []
  for i in track_direction:
    if i == "diagonal" or i == "left":
      X_match.append(X[-1])
      X = X[:-1]
    else:
      X_match.append("_")
  
  for i in track_direction:
    if i == "diagonal" or i == "up":
      Y_match.append(Y[-1])
      Y = Y[:-1]
    else:
      Y_match.append("_")
  
  x_pie = "".join(X_match[::-1])
  y_pie = "".join(Y_match[::-1])
  best_score = F[n,m]

  return x_pie, y_pie, best_score


# Following the T matrix to get the steps taken to achieve the maximum score <- recursion
def track_back(T, n, m, track_direction):
  while n!=0 or m!=0:
    if T[n, m] == 1:
      track_direction.append("diagonal")
      return track_back(T, n-1, m-1, track_direction)
    elif T[n,m] == 2:
      track_direction.append("left")
      return track_back(T, n-1, m, track_direction)
    else:
      track_direction.append("up")
      return track_back(T, n, m-1, track_direction)
  return track_direction

X = "HEAGAWGHEE"
Y = "PAWHEAE"
print(DP_GLOBAL_ALIGHMENT(X, Y))

# ('HEAGAWGHE_E', '__P_AW_HEAE', 1.0)
