// Scalar normalization used by compiled Muon plans. Matrix iterations remain SDK matmuls.
export function inverseNorm(squares, epsilon) {
    if (!Number.isFinite(squares) || squares < 0 || !Number.isFinite(epsilon) || epsilon <= 0)
        throw Error('Invalid optimizer norm');
    return 1 / (Math.sqrt(squares) + epsilon);
}
