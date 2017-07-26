/*
Copyright 2017 Google Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// This programme attempts at assigning area chairs to senior area chairs.
// It was used for NIPS 2017.
//
// You first need to prepare an ascii file that contains one line per
// constraint, as follows, where <t> means a tab:
// <sac_name> <t> <ac_name> <t> <rank> <t> <tpms>
// where sac_name is a string identifying the senior area chair,
//       ac_name is a string identifying the area chair,
//       rank is an integer that ranks how the senior area chair wants to
//            have the area chair in his set: the lower the better, and 100
//            means un-ranked,
//       tpms is a float providing the Toronto Paper Matching System similarity
//            between the senior area chair and the area chair.
// Note that contrary to the other programmes, this one assumes the input file
// does not contain a header line!
//
// We ran it with the following arguments:
//  --constraints file_containing_the_constraints
//  --results file_containing_the_results
// and the latter contains one line per assignment.

#include <string>

#include "ortools/base/commandlineflags.h"
#include "ortools/base/logging.h"
#include "ortools/base/file.h"
#include "ortools/base/map_util.h"
#include "ortools/base/numbers.h"
#include "ortools/base/split.h"
#include "ortools/base/stringprintf.h"
#include "ortools/linear_solver/linear_solver.h"

DEFINE_double(max_cost, 40, "Maximum cost");
DEFINE_string(constraints, "nips_file", "Name of file containing contraints");
DEFINE_string(results, "nips_result", "Name of file containing results");
DEFINE_int32(max_num_sacs, 25, "Maximum number of SACs");
DEFINE_int32(max_num_acs, 186, "Maximum number of ACs");
DEFINE_int32(mip_min_acs, 7, "Minimum expected ACs for MIP");
DEFINE_int32(mip_max_acs, 8, "Maximum expected ACs for MIP");
DEFINE_double(tpms_multiplier, 2.0, "TPMS score multiplier");
DEFINE_int32(max_num_at_max_cost_per_sac, 2,
    "Maximum number of unrated ACs per SAC");

namespace operations_research {

// The actual integer programming problem is prepared and solved here.
void RunIntegerProgrammingSolver(
    MPSolver::OptimizationProblemType optimization_problem_type,
    std::vector<std::vector<float> >* cost,
    std::vector<std::vector<float> >* rank,
    std::unordered_map<int32, std::string>* idx_to_sac,
    std::unordered_map<int32, std::string>* idx_to_ac,
    int32 num_sacs, int32 num_acs) {
  MPSolver solver("IntegerProgrammingExample", optimization_problem_type);

  // Assignment variables to optimize: 0 or 1.
  std::vector<std::vector<MPVariable*> > assignment;
  assignment.resize(num_sacs);
  for (int i = 0; i < num_sacs; ++i) {
    assignment[i].resize(num_acs);
    for (int j = 0; j < num_acs; ++j) {
      std::string name = StringPrintf("a_%d_%d", i, j);
      assignment[i][j] = solver.MakeIntVar(0.0, 1.0, name);
    }
  }

  // Objective to minimize: the sum of weighted assignments.
  MPObjective* const objective = solver.MutableObjective();
  for (int i = 0; i < num_sacs; ++i) {
    for (int j = 0; j < num_acs; ++j) {
      objective->SetCoefficient(assignment[i][j], (*cost)[i][j]);
    }
  }

  // Constraints: each SAC should have a bounded number of ACs assigned.
  std::vector<MPConstraint*> c0;
  c0.resize(num_sacs);
  for (int i = 0; i < num_sacs; ++i) {
    c0[i] = solver.MakeRowConstraint(FLAGS_mip_min_acs, FLAGS_mip_max_acs);
    for (int j = 0; j < num_acs; ++j) {
      c0[i]->SetCoefficient(assignment[i][j], 1.0);
    }
  }

  // Constraints: Each AC should be assigned to exactly 1 SAC.
  std::vector<MPConstraint*> c1;
  c1.resize(num_acs);
  for (int j = 0; j < num_acs; ++j) {
    c1[j] = solver.MakeRowConstraint(1.0, 1.0);
    for (int i = 0; i < num_sacs; ++i) {
      c1[j]->SetCoefficient(assignment[i][j], 1.0);
    }
  }

  // Constraints: Each SAC should not have too many "unranked" ACs.
  std::vector<MPConstraint*> c2;
  c2.resize(num_sacs);
  for (int i = 0; i < num_sacs; ++i) {
    int num_not_at_max = 0;
    for (int j = 0; j < num_acs; ++j) {
      if ((*cost)[i][j] < FLAGS_max_cost) {
        num_not_at_max++;
      }
    }
    int32 max_num_at_max_cost_per_sac = FLAGS_max_num_at_max_cost_per_sac;
    if (FLAGS_mip_max_acs - FLAGS_max_num_at_max_cost_per_sac >
        num_not_at_max) {
      max_num_at_max_cost_per_sac = FLAGS_mip_max_acs - num_not_at_max;
    }
    c2[i] = solver.MakeRowConstraint(0.0, max_num_at_max_cost_per_sac);
    for (int j = 0; j < num_acs; ++j) {
      c2[i]->SetCoefficient(assignment[i][j],
          (*cost)[i][j] >= FLAGS_max_cost ? 1.0 : 0.0);
    }
  }

  const MPSolver::ResultStatus result_status = solver.Solve();

  // Check that the problem has an optimal solution.
  if (result_status != MPSolver::OPTIMAL) {
    LOG(FATAL) << "The problem does not have an optimal solution. Status: "
               << result_status;
  }

  // The objective value of the solution.
  LOG(INFO) << "Objective value = " << objective->Value();

  std::string result = "";
  int min_num = num_acs;
  int max_num = 0;
  for (int i = 0; i < num_sacs; ++i) {
    int num = 0;
    for (int j = 0; j < num_acs; ++j) {
      if (assignment[i][j]->solution_value() > 0) {
        num++;
        result += StringPrintf("%s\t%s\t%f\t%d\n", (*idx_to_sac)[i].c_str(),
            (*idx_to_ac)[j].c_str(), (*cost)[i][j],
            static_cast<int32>((*rank)[i][j]));
      }
    }
    LOG(INFO) << "Number of ACs for " << (*idx_to_sac)[i] << " = " << num;
    if (num < min_num) {
      min_num = num;
    }
    if (num > max_num) {
      max_num = num;
    }
  }
  LOG(INFO) << "Minimum number of ACs per SAC: " << min_num;
  LOG(INFO) << "Maximum number of ACs per SAC: " << max_num;
  File* fp = File::OpenOrDie(FLAGS_results, "w");
  fp->WriteString(result);
  fp->Close();
}

}  // namespace operations_research


int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags( &argc, &argv, true);
  std::unordered_map<std::string, int32> sac;
  std::unordered_map<std::string, int32> ac;
  std::unordered_map<int32, std::string> idx_to_sac;
  std::unordered_map<int32, std::string> idx_to_ac;
  std::string all_file;
  int num_sacs = 0;
  int num_acs = 0;
  // Reading the set of constraints
  CHECK_OK(file::GetContents(FLAGS_constraints, &all_file, file::Defaults()));
  std::vector<std::string> lines =
      strings::Split(all_file, "\n", strings::SkipEmpty());
  std::vector<std::vector<float> > cost;
  std::vector<std::vector<float> > rank;
  cost.resize(FLAGS_max_num_sacs);
  rank.resize(FLAGS_max_num_sacs);
  for (int i = 0; i < FLAGS_max_num_sacs; ++i) {
    cost[i].resize(FLAGS_max_num_acs);
    rank[i].resize(FLAGS_max_num_acs);
    for (int j = 0; j < FLAGS_max_num_acs; ++j) {
      cost[i][j] = FLAGS_max_cost + FLAGS_tpms_multiplier;
      rank[i][j] = FLAGS_max_cost;
    }
  }
  for (int i = 0; i < lines.size(); ++i) {
    std::vector<std::string> fields =
        strings::Split(lines[i], "\t", strings::SkipEmpty());
    CHECK_EQ(fields.size(), 4) << "line not correct " << lines[i];
    if (!operations_research::FindOrNull(sac, fields[0])) {
      sac[fields[0]] = num_sacs;
      idx_to_sac[num_sacs] = fields[0];
      num_sacs++;
      CHECK_LE(num_sacs, FLAGS_max_num_sacs);
    }
    if (!operations_research::FindOrNull(ac, fields[1])) {
      ac[fields[1]] = num_acs;
      idx_to_ac[num_acs] = fields[1];
      num_acs++;
      CHECK_LE(num_acs, FLAGS_max_num_acs);
    }
    float value;
    CHECK(operations_research::safe_strtof(fields[2], &value));
    if (value > FLAGS_max_cost) {
      value = FLAGS_max_cost;
    }
    float tpms;
    CHECK(operations_research::safe_strtof(fields[3], &tpms));
    int32 idx_sac = sac[fields[0]];
    int32 idx_ac = ac[fields[1]];
    float score = value + (1.0 - tpms) * FLAGS_tpms_multiplier;
    if (score < 0) {
      LOG(INFO) << "score is negative: " << score;
    }
    cost[idx_sac][idx_ac] = score;
    rank[idx_sac][idx_ac] = value;
  }

  // Launching the solver
  operations_research::RunIntegerProgrammingSolver(
      // operations_research::MPSolver::SCIP_MIXED_INTEGER_PROGRAMMING,
      operations_research::MPSolver::CBC_MIXED_INTEGER_PROGRAMMING,
      &cost, &rank, &idx_to_sac, &idx_to_ac, num_sacs, num_acs);
  return 0;
}
