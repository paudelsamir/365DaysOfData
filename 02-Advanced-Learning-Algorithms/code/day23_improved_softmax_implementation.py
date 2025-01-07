model = Sequential(
    [ 
        Dense(25, activation = 'relu'),
        Dense(15, activation = 'relu'),
        Dense(4, activation = 'softmax')    # < softmax activation here

        ##         Dense(4, activation = 'linear')   #<-- Note
    ]
)
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    ##     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    # #<-- Note ---- This is preferred model softmax and loss are combined for more accurate result.
    optimizer=tf.keras.optimizers.Adam(0.001),
)

model.fit(
    X_train,y_train,
    epochs=10
)
    