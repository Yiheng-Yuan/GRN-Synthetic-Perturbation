import ch.epfl.lis.gnw.Gene;
import ch.epfl.lis.gnw.GeneNetwork;
import ch.epfl.lis.gnw.GnwSettings;
import ch.epfl.lis.gnw.HillGene;
import ch.epfl.lis.gnw.PerturbationSingleGene;
import ch.epfl.lis.gnw.Solver;
import ch.epfl.lis.networks.Edge;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;

/** Acceptance driver using the unmodified official GNW JAR, not a replacement simulator. */
public final class GnwAcceptance {
    // Rows are regulated genes, columns are regulators. Six edges include a
    // feed-forward loop, positive G2/G3 feedback and negative G3/G4 feedback.
    private static final int[][] SIGN = {
        {0, 0, 0, 0}, {1, 0, 1, 0}, {1, 1, 0, -1}, {0, 0, 1, 0}
    };
    private static final double[] RHO = {0.8, 0.85, 0.9, 0.75};
    private static final double[] DOSES = {0.3, 0.5, 0.7, 0.9};
    private static final int N = 4;
    private static final double BASAL = 0.4;
    private static final double EFFECT = 0.12;
    private static final double K = 0.5;

    private static String label(int index) { return index < 0 ? "ctrl" : "G" + (index + 1); }

    private static void parameter(ArrayList<String> names, ArrayList<Double> values,
                                  String name, double value) {
        names.add(name);
        values.add(value);
    }

    private static GeneNetwork network(int clockScale, double tolerance) {
        GnwSettings settings = GnwSettings.getInstance();
        settings.setModelTranslation(false); // RNA-only, no unobserved protein state.
        settings.setDt(1); // GNW's output clock is integer-valued.
        settings.setAbsolutePrecision(tolerance);
        settings.setRelativePrecision(tolerance);
        settings.setRandomSeed(20261001);
        settings.setAddNormalNoise(false);
        settings.setAddLognormalNoise(false);
        settings.setAddMicroarrayNoise(false);
        settings.setNormalizeAfterAddingNoise(false);
        settings.setSimulateODE(true);
        settings.setSimulateSDE(false);
        GeneNetwork grn = new GeneNetwork();
        grn.setId("gnw_acceptance_four_genes");
        grn.setDirected(true);
        grn.setSigned(true);
        for (int i = 0; i < N; i++) {
            HillGene gene = new HillGene(grn);
            gene.setLabel(label(i));
            grn.addNode(gene);
        }
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (SIGN[i][j] != 0) {
                    grn.addEdge(new Edge(grn.getNode(j), grn.getNode(i),
                        SIGN[i][j] > 0 ? Edge.ENHANCER : Edge.INHIBITOR));
                }
            }
        }
        grn.setSize(N);
        for (int i = 0; i < N; i++) {
            ArrayList<Gene> inputs = new ArrayList<>();
            ArrayList<Integer> signs = new ArrayList<>();
            ArrayList<String> names = new ArrayList<>();
            ArrayList<Double> values = new ArrayList<>();
            // tau = clockScale*t; d x/d tau = (d x/d t)/clockScale.
            parameter(names, values, "max", 1.0 / clockScale);
            parameter(names, values, "delta", 1.0 / clockScale);
            for (int j = 0; j < N; j++) {
                if (SIGN[i][j] == 0) continue;
                inputs.add(grn.getGene(j));
                signs.add(SIGN[i][j]);
                int module = inputs.size();
                parameter(names, values, "bindsAsComplex_" + module, 0);
                parameter(names, values, "numActivators_" + module, 1);
                parameter(names, values, "numDeactivators_" + module, 0);
                parameter(names, values, "k_" + module, K);
                parameter(names, values, "n_" + module, 1);
            }
            for (int state = 0; state < (1 << inputs.size()); state++) {
                double alpha = BASAL;
                for (int m = 0; m < inputs.size(); m++) {
                    if ((state & (1 << m)) != 0) alpha += EFFECT * signs.get(m);
                }
                if (!(alpha >= 0 && alpha <= 1)) throw new AssertionError("Invalid alpha");
                parameter(names, values, "a_" + state, alpha);
            }
            grn.getGene(i).initialization(names, values, inputs);
        }
        grn.setEdgeTypesAccordingToDynamicalModel();
        return grn;
    }

    private static PerturbationSingleGene perturb(GeneNetwork grn, int target,
                                                 double dose, double rho) {
        if (target < 0) return null;
        if (dose < 0 || rho < 0 || dose * rho > 1) throw new IllegalArgumentException();
        PerturbationSingleGene p = new PerturbationSingleGene(grn);
        p.singleGenePerturbations(1 - rho * dose);
        p.applyPerturbation(target);
        return p;
    }

    private static double[] rhs(GeneNetwork grn, double[] state, int scale) {
        double[] out = new double[N];
        grn.computeDxydt(state, out);
        for (int i = 0; i < N; i++) out[i] *= scale;
        return out;
    }

    private static double maxAbs(double[] x) {
        double result = 0;
        for (double v : x) result = Math.max(result, Math.abs(v));
        return result;
    }

    private static void legal(double[] x) {
        for (double v : x) {
            // Deliberately do not clip concentrations to pass the check.
            if (!Double.isFinite(v) || v < 0) throw new AssertionError("Illegal state " + v);
        }
    }

    private static void step(Solver solver) throws Exception {
        double advanced = solver.step();
        if (Math.abs(advanced - 1) > 1e-10) throw new AssertionError("Clock advance " + advanced);
        legal(solver.getState());
    }

    private static double[] steady(GeneNetwork grn, double[] initial, int scale,
                                   int target, double dose, double rho,
                                   PrintWriter output, String initialId) throws Exception {
        PerturbationSingleGene p = perturb(grn, target, dose, rho);
        try {
            Solver solver = new Solver(Solver.type.ODE, grn, initial.clone());
            int consecutive = 0;
            for (int tick = 1; tick <= 256 * scale; tick++) {
                step(solver);
                double[] x = solver.getState().clone();
                double residual = maxAbs(rhs(grn, x, scale));
                consecutive = residual < 1e-11 ? consecutive + 1 : 0;
                if (consecutive >= 8) {
                    if (output != null) {
                        output.print(initialId + "," + label(target) + "," + dose + ","
                            + ((double) tick / scale) + "," + residual);
                        vector(output, x);
                        output.println();
                    }
                    return x;
                }
            }
            throw new AssertionError("No converged steady state");
        } finally {
            if (p != null) p.restoreWildType();
        }
    }

    private static void vector(PrintWriter out, double[] x) {
        for (double v : x) out.print("," + Double.toString(v));
    }

    private static PrintWriter writer(Path path) throws Exception {
        return new PrintWriter(Files.newBufferedWriter(path, StandardCharsets.UTF_8));
    }

    private static void trajectory(GeneNetwork grn, double[] initial, int initialId,
                                   int target, double dose, double rho, int scale,
                                   PrintWriter out, String runId) throws Exception {
        // Store the pre-intervention t=0, then hold the production intervention
        // for the whole trajectory. No hidden prerun or halfway restoration.
        String prefix = runId + "," + initialId + "," + label(target) + "," + dose + ","
            + rho + "," + (target < 0 ? 1.0 : 1 - rho * dose);
        out.print(prefix + ",0.0");
        vector(out, initial);
        out.println();
        PerturbationSingleGene p = perturb(grn, target, dose, rho);
        try {
            Solver solver = new Solver(Solver.type.ODE, grn, initial.clone());
            int outputStride = scale / 8;
            for (int tick = 1; tick <= 8 * scale; tick++) {
                step(solver);
                if (tick % outputStride != 0) continue;
                out.print(prefix + "," + ((double) tick / scale));
                vector(out, solver.getState());
                out.println();
                for (int i = 0; i < N; i++) {
                    double expectedMax = (i == target ? 1 - rho * dose : 1) / scale;
                    if (Math.abs(grn.getGene(i).getMax() - expectedMax) > 1e-15
                        || Math.abs(grn.getGene(i).getDelta() - 1.0 / scale) > 1e-15)
                        throw new AssertionError("Intervention did not persist");
                }
            }
        } finally {
            if (p != null) p.restoreWildType();
        }
    }

    private static void trajectories(Path path, int scale, double tolerance,
                                     double[][] initials) throws Exception {
        GeneNetwork grn = network(scale, tolerance);
        try (PrintWriter out = writer(path)) {
            out.println("run_id,initial_id,target,dose,rho,remaining_fraction,time,G1,G2,G3,G4");
            for (int initial = 0; initial < initials.length; initial++) {
                trajectory(grn, initials[initial], initial, -1, 0, 0, scale, out, "i" + initial + "_ctrl");
                for (int q = 0; q < N; q++) {
                    for (double dose : DOSES) {
                        trajectory(grn, initials[initial], initial, q, dose, RHO[q], scale, out,
                            "i" + initial + "_" + label(q) + "_d" + dose);
                    }
                }
            }
            trajectory(grn, initials[0], 0, 0, 0, RHO[0], scale, out, "zero_dose");
            trajectory(grn, initials[0], 0, 0, 1, 1, scale, out, "complete_ko");
        }
    }

    private static void exportTruth(Path directory, GeneNetwork grn, int scale) throws Exception {
        Files.createDirectories(directory);
        try (PrintWriter out = writer(directory.resolve("parameters.json"))) {
            out.println("{\"genes\":[\"G1\",\"G2\",\"G3\",\"G4\"],");
            out.println("\"signed_matrix\":" + Arrays.deepToString(SIGN) + ",");
            out.println("\"coefficient_matrix\":[[0,0,0,0],[0.12,0,0.12,0],[0.12,0.12,0,-0.12],[0,0,0.12,0]],");
            out.println("\"k\":0.5,\"n\":1,\"basal\":0.4,\"max_production\":1,\"decay\":1,");
            out.println("\"rho\":" + Arrays.toString(RHO) + ",\"clock_scale\":" + scale + ",");
            out.println("\"model_translation\":false,\"process_noise\":false,\"state_clipping\":false,");
            out.println("\"jacobian_state\":[0.4,0.5,0.6,0.7],\"jacobian_difference_step\":1e-6,");
            out.println("\"gene_parameters\":{");
            for (int i = 0; i < N; i++) {
                ArrayList<String> names = new ArrayList<>();
                ArrayList<Double> values = new ArrayList<>();
                Gene gene = grn.getGene(i);
                gene.compileParameters(names, values); // Export actual native parameters.
                out.print("\"" + label(i) + "\":{\"inputs\":[");
                for (int m = 0; m < gene.getInputGenes().size(); m++) {
                    if (m > 0) out.print(",");
                    out.print("\"" + gene.getInputGenes().get(m).getLabel() + "\"");
                }
                out.print("],\"parameters\":{");
                for (int m = 0; m < names.size(); m++) {
                    if (m > 0) out.print(",");
                    out.print("\"" + names.get(m) + "\":" + values.get(m));
                }
                out.println("}}" + (i == N - 1 ? "" : ","));
            }
            out.println("}}");
        }
        try (PrintWriter out = writer(directory.resolve("jacobian.csv"))) {
            double[] x = {0.4, 0.5, 0.6, 0.7};
            out.println("target,G1,G2,G3,G4");
            double[][] jacobian = new double[N][N];
            for (int j = 0; j < N; j++) {
                double[] plus = x.clone(), minus = x.clone();
                plus[j] += 1e-6;
                minus[j] -= 1e-6;
                double[] fp = rhs(grn, plus, scale), fm = rhs(grn, minus, scale);
                for (int i = 0; i < N; i++) jacobian[i][j] = (fp[i] - fm[i]) / 2e-6;
            }
            for (int i = 0; i < N; i++) {
                out.print(label(i)); vector(out, jacobian[i]); out.println();
            }
        }
    }

    private static void probes(Path path, GeneNetwork grn, int scale) throws Exception {
        double[][] states = {{0, 0, 0, 0}, {0.05, 0.1, 0.2, 0.3},
            {0.4, 0.5, 0.6, 0.7}, {1, 2, 3, 4}};
        try (PrintWriter out = writer(path)) {
            out.println("probe_id,target,dose,rho,x1,x2,x3,x4,dx1,dx2,dx3,dx4,production1,production2,production3,production4");
            for (int probe = 0; probe < states.length; probe++) {
                for (int q = -1; q < N; q++) {
                    double[] doses = q < 0 ? new double[]{0} : new double[]{0, 0.3, 0.7, 0.9};
                    for (double dose : doses) {
                        double rho = q < 0 ? 0 : RHO[q];
                        PerturbationSingleGene p = perturb(grn, q, dose, rho);
                        try {
                            out.print(probe + "," + label(q) + "," + dose + "," + rho);
                            vector(out, states[probe]);
                            vector(out, rhs(grn, states[probe], scale));
                            double[] production = new double[N];
                            for (int i = 0; i < N; i++) production[i] = scale
                                * grn.getGene(i).computeMRnaProductionRate(i, new DenseDoubleMatrix1D(states[probe]));
                            vector(out, production);
                            out.println();
                        } finally {
                            if (p != null) p.restoreWildType();
                        }
                    }
                }
            }
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 1) throw new IllegalArgumentException("Usage: GnwAcceptance OUTPUT_DIRECTORY");
        Path output = Path.of(args[0]);
        Files.createDirectories(output);
        final int scale = 8;
        GeneNetwork grn = network(scale, 1e-10);
        exportTruth(output.resolve("truth"), grn, scale);
        try (PrintWriter edges = writer(output.resolve("edges.csv"))) {
            edges.println("regulator,target,sign");
            for (Edge edge : grn.getEdges()) {
                int sign = edge.getType().equals(Edge.ENHANCER) ? 1
                    : edge.getType().equals(Edge.INHIBITOR) ? -1 : 0;
                if (sign == 0) throw new AssertionError("Unsigned edge");
                edges.println(edge.getSource().getLabel() + "," + edge.getTarget().getLabel() + "," + sign);
            }
        }
        probes(output.resolve("probes.csv"), grn, scale);
        double[] baseline;
        try (PrintWriter out = writer(output.resolve("steady_states.csv"))) {
            out.println("initial_id,target,dose,time,residual,G1,G2,G3,G4");
            baseline = steady(grn, new double[]{0.4, 0.4, 0.4, 0.4}, scale, -1, 0, 0, out, "baseline");
            double[][] seeds = {{0, 0, 0, 0}, {0.01, 0.1, 0.2, 0.3},
                {1, 1, 1, 1}, {10, 10, 10, 10}, {1, 0.02, 4, 0.5}};
            for (int initial = 0; initial < seeds.length; initial++) {
                steady(grn, seeds[initial], scale, -1, 0, 0, out, "seed" + initial);
            }
            for (int q = 0; q < N; q++) {
                for (double dose : DOSES) {
                    steady(grn, baseline, scale, q, dose, RHO[q], out, "baseline");
                    // Also check two widely separated starts for every perturbation.
                    steady(grn, seeds[0], scale, q, dose, RHO[q], out, "seed0");
                    steady(grn, seeds[3], scale, q, dose, RHO[q], out, "seed3");
                }
            }
        }
        double[] varied1 = baseline.clone(), varied2 = baseline.clone();
        double[] factors = {0.90, 1.10, 0.95, 1.05};
        for (int i = 0; i < N; i++) {
            varied1[i] *= factors[i];
            varied2[i] *= 2 - factors[i];
        }
        double[][] initials = {baseline, varied1, varied2};
        trajectories(output.resolve("trajectories.csv"), 8, 1e-10, initials);
        trajectories(output.resolve("refined_trajectories.csv"), 16, 1e-12, initials);
        System.out.println("Native GNW " + GnwSettings.getInstance().getGnwVersion()
            + " acceptance generation complete: 4 genes, 6 edges, 53 trajectories x 65 times.");
        System.out.println("Output: " + output.toAbsolutePath());
    }
}
