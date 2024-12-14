import numpy as np
import conllu


def viterbi(obs, states, pi, A, b):
    N = len(states)
    T = len(obs)
    # Initialize a path probability matrix
    viterbi = np.zeros((N, T))
    backpointer = np.zeros((N, T), dtype=int)

    # Initialization step
    for q in range(N):
        if obs[0] in b[states[q]].keys():
            viterbi[q, 0] = pi[states[q]] * b[states[q]][obs[0]]

    # Recursion step
    for t in range(1, T):
        for q in range(N):
            '''maxProb, bestState = max(
                (viterbi[q_prime, t - 1] * A[states[q_prime]][states[q]] * b[states[q]][obs[t]], q_prime)
                for q_prime in range(N) 
            )'''
            maxProb = 0
            bestState = 0
            for q_prime in range(N):
                if states[q] in A[states[q_prime]].keys() and obs[t] in b[states[q]].keys():
                    if viterbi[q_prime, t - 1] * A[states[q_prime]][states[q]] * b[states[q]][obs[t]] > maxProb:
                        maxProb = viterbi[q_prime, t - 1] * A[states[q_prime]][states[q]] * b[states[q]][obs[t]]
                        bestState = q_prime
            viterbi[q, t] = maxProb
            backpointer[q, t] = bestState

    '''bestPathProb, bestLastState = max(
                (viterbi[q_prime, T - 1] * A[states[q_prime]]['STOP'], q_prime)
                for q_prime in range(N)
            )'''
    bestPathProb = 0
    bestLastState = 0
    for q_prime in range(N):
        if 'STOP' in A[states[q_prime]].keys():
            if viterbi[q_prime, T - 1] * A[states[q_prime]]['STOP'] > bestPathProb:
                bestPathProb = viterbi[q_prime, T - 1] * A[states[q_prime]]['STOP']
                bestLastState = q_prime

    bestPath = [bestLastState]
    for t in range(T - 1, 0, -1):
        bestPath.append(backpointer[bestPath[-1], t])
    bestPath.reverse()

    return bestPathProb, bestPath


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

        first_tag = pos_tags[0]
        if first_tag in pi:
            pi[first_tag] += 1
        else:
            pi[first_tag] = 1

        prev = first_tag
        for tag in pos_tags[1:]:
            if prev in A:
                if tag in A[prev]:
                    A[prev][tag] += 1
                else:
                    A[prev][tag] = 1
            else:
                A[prev] = {}
                A[prev][tag] = 1
            prev = tag
        if prev in A:
            if 'STOP' in A[prev]:
                A[prev]['STOP'] += 1
            else:
                A[prev]['STOP'] = 1
        else:
            A[prev] = {}
            A[prev]['STOP'] = 1

        for i in range(len(tokens)):
            if pos_tags[i] in b:
                if tokens[i] in b[pos_tags[i]]:
                    b[pos_tags[i]][tokens[i]] += 1
                else:
                    b[pos_tags[i]][tokens[i]] = 1
            else:
                b[pos_tags[i]] = {}
                b[pos_tags[i]][tokens[i]] = 1


    num = len(parsed_data)
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
    overscore = 0

    for sentence in parsed_data:
        tokens = [token["form"] for token in sentence]  # Get tokens
        pos_tags = [token["upos"] for token in sentence]  # Get POS tags

        _, path = viterbi(tokens, tags, pi, A, b)
        new_tags = []
        for i in path:
            new_tags.append(tags[i])

        score = 0
        assert len(pos_tags) == len(new_tags)
        for i in range(len(pos_tags)):
            if pos_tags[i] == new_tags[i]:
                score += 1
        overscore += score / len(pos_tags)
    return overscore / len(parsed_data)


if __name__ == "__main__":
    pi, A, b = train("en_gum-ud-train.conllu")
    tags = list(b.keys())
    accuracy = evaluate("en_gum-ud-test.conllu", tags, pi, A, b)

    print(accuracy)
