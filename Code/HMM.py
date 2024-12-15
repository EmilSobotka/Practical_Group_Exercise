import numpy as np
import conllu


import numpy as np
from collections import defaultdict

def viterbi(obs, states, pi, A, b):
    N = len(states)  # Number of states
    T = len(obs)     # Length of observation sequence

    # Initialize the path probability matrix and backpointer
    viterbi = np.zeros((N, T))
    backpointer = np.zeros((N, T), dtype=int)

    # Initialization step: t = 0
    for q in range(N):
        # Handle cases where observation or state probabilities may be missing
        viterbi[q, 0] = pi.get(states[q], 0) * b.get(states[q], {}).get(obs[0], 0)

    # Recursion step: t = 1 to T-1
    for t in range(1, T):
        for q in range(N):
            max_prob = 0
            best_state = 0
            for q_prime in range(N):
                # Ensure all probabilities exist, using 0 for missing probabilities
                transition_prob = A.get(states[q_prime], {}).get(states[q], 0)
                emission_prob = b.get(states[q], {}).get(obs[t], 0)
                prob = viterbi[q_prime, t - 1] * transition_prob * emission_prob

                if prob > max_prob:
                    max_prob = prob
                    best_state = q_prime

            viterbi[q, t] = max_prob
            backpointer[q, t] = best_state

    # Termination step
    best_path_prob = 0
    best_last_state = 0
    for q in range(N):
        stop_prob = A.get(states[q], {}).get('STOP', 0)
        prob = viterbi[q, T - 1] * stop_prob

        if prob > best_path_prob:
            best_path_prob = prob
            best_last_state = q

    # Reconstruct the best path
    best_path = [best_last_state]
    for t in range(T - 1, 0, -1):
        best_path.append(backpointer[best_path[-1], t])
    best_path.reverse()

    return best_path_prob, best_path

def train(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        data = f.read()

    # Parse the file
    parsed_data = conllu.parse(data)
    pi = {}
    A = {}
    b = {}

    # Extract tokens and POS tags
    for sentence in parsed_data:
        tokens = [token["form"] for token in sentence]  # Get tokens
        pos_tags = [token["upos"] for token in sentence]  # Get POS tags

        first_tag = pos_tags[0] #Count first tags
        if first_tag in pi:
            pi[first_tag] += 1
        else:
            pi[first_tag] = 1

        prev = first_tag
        for tag in pos_tags[1:]: #Count Transitions
            if prev in A:
                if tag in A[prev]:
                    A[prev][tag] += 1
                else:
                    A[prev][tag] = 1
            else:
                A[prev] = {}
                A[prev][tag] = 1
            prev = tag
        if prev in A: #Count Endings
            if 'STOP' in A[prev]:
                A[prev]['STOP'] += 1
            else:
                A[prev]['STOP'] = 1
        else:
            A[prev] = {}
            A[prev]['STOP'] = 1

        for i in range(len(tokens)): #Counts for emission matrix
            if pos_tags[i] in b:
                if tokens[i] in b[pos_tags[i]]:
                    b[pos_tags[i]][tokens[i]] += 1
                else:
                    b[pos_tags[i]][tokens[i]] = 1
            else:
                b[pos_tags[i]] = {}
                b[pos_tags[i]][tokens[i]] = 1


    num = len(parsed_data) #Transform to probabilities
    for i in pi:
        pi[i] = pi[i] / num

    for j in A:
        num = sum(A[j].values())
        for i in A[j]:
            A[j][i] = A[j][i] / num

    for j in b:
        num = sum(b[j].values())
        for i in b[j]:
            b[j][i] = b[j][i] / num

    '''try:
        print(b)
    except UnicodeEncodeError as e:
        print(f"Unicode error: {e}")
        print("Attempting to identify problematic tokens...")
        for key, value in b.items():
            try:
                print(f"{key}: {value}")
            except UnicodeEncodeError:
                print(f"Problematic key: {key}")'''

    return pi, A, b

def evaluate(file_name, tags, pi, A, b):
    with open(file_name, "r", encoding="utf-8") as f:
        data = f.read()

    parsed_data = conllu.parse(data)
    overscore = 0 #This will accumulate the overall score across all sentences

    for sentence in parsed_data:
        tokens = [token["form"] for token in sentence]  # Get tokens
        pos_tags = [token["upos"] for token in sentence]  # Get POS tags

        _, path = viterbi(tokens, tags, pi, A, b) #Run the Viterbi algorithm to predict the most likely POS tag sequence
        new_tags = []
        for i in path:
            new_tags.append(tags[i])

        score = 0 #Score to track how many tags match the true labels
        assert len(pos_tags) == len(new_tags)

        #Compare the predicted tags with the true tags
        for i in range(len(pos_tags)):
            if pos_tags[i] == new_tags[i]:
                score += 1
        overscore += score / len(pos_tags)
    return overscore / len(parsed_data)

def error_analysis(file_name, train_file_name):
    # Train the HMM model on the training file and get the probability matrices
    pi, A, b = train(train_file_name)

    tags = list(b.keys())

    with open(file_name, "r", encoding="utf-8") as f:
        data = f.read()

    parsed_data = conllu.parse(data)

    # Dictionary to store substitution errors and previous tags
    substitution_errors = {}

    # Process each sentence in the test data
    for sentence in parsed_data:
        tokens = [token["form"] for token in sentence]  # Get tokens
        pos_tags = [token["upos"] for token in sentence]  # Get true POS tags

        # Run Viterbi to predict tags
        _, path = viterbi(tokens, tags, pi, A, b)
        predicted_tags = [tags[i] for i in path]

        # Track errors
        for i, (true_tag, pred_tag) in enumerate(zip(pos_tags, predicted_tags)):
            if true_tag != pred_tag:
                # Track substitution errors along with the previous tag
                if i > 0:
                    previous_tag = pos_tags[i - 1]
                else:
                    previous_tag = '*'  # Start-of-sentence marker

                # Store the substitution error with the previous tag
                if (previous_tag, true_tag, pred_tag) not in substitution_errors:
                    substitution_errors[(previous_tag, true_tag, pred_tag)] = 1
                else:
                    substitution_errors[(previous_tag, true_tag, pred_tag)] += 1

    # Sort substitution errors by the number of occurrences
    sorted_substitution_errors = sorted(substitution_errors.items(), key=lambda x: x[1], reverse=True)

    # Print the 5 most common substitution errors and their probabilities
    print("\nTop 5 Most Common Substitution Errors (Prev Tag -> True Tag -> Predicted Tag):")
    for (prev_tag, true_tag, pred_tag), count in sorted_substitution_errors[:5]:
        # Transition probabilities
        true_to_true_prob = A.get(prev_tag, {}).get(true_tag, 0)
        prev_to_pred_prob = A.get(prev_tag, {}).get(pred_tag, 0)

        # Obtain the current token to calculate the emission probability
        # Ensure that a match exists for the true_tag before accessing token_index
        matching_indices = [i for i, t in enumerate(pos_tags) if t == true_tag]
        if matching_indices:
            token_index = matching_indices[0]
            emission_prob = b.get(true_tag, {}).get(tokens[token_index], 0)
        else:
            # Handle the case where no matching token is found
            token_index = None
            emission_prob = 0  # Default emission probability in case of error

        # Calculate transition probability multiplied by emission probability
        transition_true_times_emission_prob = true_to_true_prob * emission_prob
        transition_pred_times_emission_prob = prev_to_pred_prob * emission_prob
         
        # Print substitution error with the original and multiplied transition probabilities
        print(f" {prev_tag} -> {true_tag} -> {pred_tag}: {count} errors")
        print(f"Transition probability from {prev_tag} to {true_tag}: {true_to_true_prob}")
        print(f"Transition probability from {prev_tag} to {pred_tag}: {prev_to_pred_prob}")
        print(f"Emission probability weight: {emission_prob}")
        print(f"True Transition Prob * Emission probability from {prev_tag} to {true_tag}: {transition_true_times_emission_prob}")
        print(f"Pred Transition Prob * Emission probability from {prev_tag} to {pred_tag}: {transition_pred_times_emission_prob}")

def apply_hmm_to_new_dataset(new_dataset_file, train_file_name):
     # Train the HMM model using the original training data
    pi, A, b = train(train_file_name)
    tags = list(b.keys())

    # Load and parse the new dataset
    with open(new_dataset_file, "r", encoding="utf-8") as f:
        new_data = f.read()
    parsed_new_data = conllu.parse(new_data)

    # Process each sentence in the new dataset
    for sentence in parsed_new_data:
        tokens = [token["form"] for token in sentence]  # Get tokens

        # Apply the Viterbi algorithm to predict POS tags
        _, predicted_path = viterbi(tokens, tags, pi, A, b)
        predicted_tags = [tags[i] for i in predicted_path]

        # Add predicted POS tags to the sentence
        for i, token in enumerate(sentence):
            token["predicted_upos"] = predicted_tags[i]
