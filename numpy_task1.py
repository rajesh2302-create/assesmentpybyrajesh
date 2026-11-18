import numpy as np
marks=np.array([[10,20,30],
                 [40,50,60],
                 [50,70,90],
                 [50,30,80]])
print("sliced 1st 2 student of last subject mark")
for num in marks[0:2,-1]:
    print (num)
students_average=np.mean(marks,axis=1)
subjects_average=np.mean(marks,axis=0)    
print(f"student average={np.round(students_average,2)}")
print(f"subject average={np.round(subjects_average,2)}")
#grace marks
grace_marks=marks+5
print("\n--- after adding grace marks 5(broadcasting)--")
print(grace_marks)
#weighted score
weights=np.array([0.3,0.3,0.4])
weighted_scores=marks @ weights
print(f"\n--weighted scores{np.round(weighted_scores,2)}")
print(f"shape{marks.shape}")
print(f"datatype{marks.dtype}")
print(f"size{marks.size}")
#reshape
new_array=marks.reshape(12)
print(new_array)
