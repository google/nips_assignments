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

// This programme attempts at assigning papers to area chairs. It was
// used for NIPS 2017.
//
// You first need to prepare an ascii file that contains one line per
// constraint, as follows, where <t> means a tab:
// ac_name <t> paper(s) <t> num_papers <t> conflict <t> bid_not_willing <t>
// bid_not_entered <t> bid_in_a_pinch <t> bid_willing <t> bid_eager <t>
// tpms <t> cosine <t> containment
// where ac_name is a string related to the area chair
//       paper(s) is a string identifying a set of papers (often one, but
//                sometimes papers need to be assigned to the same AC...)
//       num_papers is an int (usually 1)
//       conflict is 1 if there is a conflict between AC and paper, 0 otherwise
//       bid_XXX is the number of papers in the set for which the AC has bid
//               XXX (eager, not willing, etc).
//       tpms is a float giving the similarity between the paper and the AC
//            as estimated by the Toronto Paper Matching System
//       cosine is the cosine similarity between paper and AC according to
//              their common topic areas.
//       containment is the containment similarity between paper and AC
//              according to their common topic areas.
// Note that the first line of this file is not read as it is assumed to be
// a header.
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

DEFINE_double(max_cost, 110, "Maximum cost");
DEFINE_string(constraints, "assignments.tsv",
    "Name of file containing contraints");
DEFINE_string(results, "ac_result", "Name of file containing results");
DEFINE_int32(max_num_acs, 186, "Maximum number of ACs");
DEFINE_int32(max_num_papers, 3600, "Maximum number of papers");
DEFINE_int32(max_not_bid_for, 10,
    "Maximum number of not-bid-for papers per AC");
DEFINE_int32(mip_min_papers, 17, "Minimum expected papers for MIP");
DEFINE_int32(mip_max_papers, 19, "Maximum expected papers for MIP");
DEFINE_double(tpms_multiplier, 1.0, "TPMS score multiplier");
DEFINE_double(conflict_multiplier, 500.0, "Conflict multiplier");
DEFINE_double(bid_multiplier, 8.0, "Bid multiplier");
DEFINE_double(not_entered_bid_multiplier, 1.0, "Not entered bid multiplier");
DEFINE_double(negative_bid_multiplier, 10.0, "Negative bid multiplier");
DEFINE_double(subject_cosine_multiplier, 0.0, "Subject cosine multiplier");
DEFINE_double(subject_containment_multiplier, 1.0,
    "Subject containment multiplier");

namespace operations_research {

// This launches the actual mixed integer programming solver
void RunIntegerProgrammingSolver(
    MPSolver::OptimizationProblemType optimization_problem_type,
    std::vector<std::vector<float> >* cost,
    std::vector<std::vector<float> >* tpms,
    std::vector<std::vector<int32> >* bids_eager,
    std::vector<std::vector<int32> >* bids_willing,
    std::vector<std::vector<int32> >* bids_in_a_pinch,
    std::vector<std::vector<int32> >* bids_not_entered,
    std::vector<std::vector<int32> >* bids_not_willing,
    std::vector<std::vector<float> >* cosine,
    std::vector<std::vector<float> >* containment,
    std::vector<float>* weight,
    std::unordered_map<int32, std::string>* idx_to_ac,
    std::unordered_map<int32, std::string>* idx_to_paper,
    int32 num_acs, int32 num_papers) {
  MPSolver solver("IntegerProgrammingSolver", optimization_problem_type);

  // Assignment variables to optimize: 0 or 1.
  std::vector<std::vector<MPVariable*> > assignment;
  assignment.resize(num_acs);
  for (int i = 0; i < num_acs; ++i) {
    assignment[i].resize(num_papers);
    for (int j = 0; j < num_papers; ++j) {
      std::string name = StringPrintf("a_%d_%d", i, j);
      assignment[i][j] = solver.MakeIntVar(0.0, 1.0, name);
    }
  }

  // Objective to minimize: the sum of weighted assignments.
  MPObjective* const objective = solver.MutableObjective();
  for (int i = 0; i < num_acs; ++i) {
    for (int j = 0; j < num_papers; ++j) {
      objective->SetCoefficient(assignment[i][j], (*cost)[i][j]);
    }
  }

  // Constraints: each AC should have a bounded number of papers assigned.
  std::vector<MPConstraint*> c0;
  c0.resize(num_acs);
  for (int i = 0; i < num_acs; ++i) {
    c0[i] = solver.MakeRowConstraint(FLAGS_mip_min_papers,
        FLAGS_mip_max_papers);
    for (int j = 0; j < num_papers; ++j) {
      c0[i]->SetCoefficient(assignment[i][j], (*weight)[j]);
    }
  }

  // Constraints: Each paper should be assigned to exactly 1 AC.
  std::vector<MPConstraint*> c1;
  c1.resize(num_papers);
  for (int j = 0; j < num_papers; ++j) {
    c1[j] = solver.MakeRowConstraint(1.0, 1.0);
    for (int i = 0; i < num_acs; ++i) {
      c1[j]->SetCoefficient(assignment[i][j], 1.0);
    }
  }

  // Constraints: Each AC should not have too many assigned papers
  // they have not asked actively for
  std::vector<MPConstraint*> c2;
  c2.resize(num_acs);
  for (int i = 0; i < num_acs; ++i) {
    c2[i] = solver.MakeRowConstraint(0.0, FLAGS_max_not_bid_for);
    for (int j = 0; j < num_papers; ++j) {
      c2[i]->SetCoefficient(assignment[i][j],
          (*bids_not_entered)[i][j] + (*bids_not_willing)[i][j]);
    }
  }


  LOG(INFO) << "Launching solver";

  const MPSolver::ResultStatus result_status = solver.Solve();

  // Check that the problem has an optimal solution.
  if (result_status != MPSolver::OPTIMAL) {
    LOG(FATAL) << "The problem does not have an optimal solution. Status: "
               << result_status;
  }

  // The objective value of the solution.
  LOG(INFO) << "Objective value = " << objective->Value();

  int min_num = num_papers;
  int max_num = 0;
  std::string result = StringPrintf("%s%s%s",
      "AC\tPaper(s)\tCost\tBidEager\tBidWilling\t",
      "BidInAPinch\tBidNotEntered\tBidNotWilling\tTPMS\t",
      "CosineSubjectSimilarity\tContainmentSubjectSimilarity\tHappiness\n");
  for (int i = 0; i < num_acs; ++i) {
    int num = 0;
    float num_negative_bids = 0;
    float num_not_entered_bids = 0;
    for (int j = 0; j < num_papers; ++j) {
      if (assignment[i][j]->solution_value() > 0) {
        num += (*weight)[j];
        num_negative_bids += (*bids_not_willing)[i][j];
        num_not_entered_bids += (*bids_not_entered)[i][j] *
          FLAGS_not_entered_bid_multiplier / FLAGS_negative_bid_multiplier;
      }
    }
    for (int j = 0; j < num_papers; ++j) {
      if (assignment[i][j]->solution_value() > 0) {
        float happiness = 1.0 -
            (num_negative_bids + num_not_entered_bids) / num;
        result += StringPrintf(
            "%s\t%s\t%f\t%d\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t%f\n",
            (*idx_to_ac)[i].c_str(),
            (*idx_to_paper)[j].c_str(), (*cost)[i][j], (*bids_eager)[i][j],
            (*bids_willing)[i][j], (*bids_in_a_pinch)[i][j],
            (*bids_not_entered)[i][j], (*bids_not_willing)[i][j],
            (*tpms)[i][j], (*cosine)[i][j], (*containment)[i][j], happiness);
      }
    }
    LOG(INFO) << "Number of papers for " << (*idx_to_ac)[i] << " = " << num;
    if (num < min_num) {
      min_num = num;
    }
    if (num > max_num) {
      max_num = num;
    }
  }
  LOG(INFO) << "Minimum number of papers per AC: " << min_num;
  LOG(INFO) << "Maximum number of papers per AC: " << max_num;
  File* fp = File::OpenOrDie(FLAGS_results, "w");
  fp->WriteString(result);
  fp->Close();
}

}  // namespace operations_research


int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags( &argc, &argv, true);
  std::unordered_map<std::string, int32> ac;
  std::unordered_map<std::string, int32> paper;
  std::unordered_map<int32, std::string> idx_to_ac;
  std::unordered_map<int32, std::string> idx_to_paper;
  std::string all_file;
  int num_acs = 0;
  int num_papers = 0;
  float total_weight = 0;
  // Reading all the constraints
  CHECK_OK(file::GetContents(FLAGS_constraints, &all_file, file::Defaults()));
  std::vector<std::string> lines =
      strings::Split(all_file, "\n", strings::SkipEmpty());
  std::vector<std::vector<float> > cost;
  cost.resize(FLAGS_max_num_acs);
  std::vector<std::vector<float> > tpms;
  tpms.resize(FLAGS_max_num_acs);
  std::vector<std::vector<int32> > bids_not_willing;
  bids_not_willing.resize(FLAGS_max_num_acs);
  std::vector<std::vector<int32> > bids_not_entered;
  bids_not_entered.resize(FLAGS_max_num_acs);
  std::vector<std::vector<int32> > bids_in_a_pinch;
  bids_in_a_pinch.resize(FLAGS_max_num_acs);
  std::vector<std::vector<int32> > bids_willing;
  bids_willing.resize(FLAGS_max_num_acs);
  std::vector<std::vector<int32> > bids_eager;
  bids_eager.resize(FLAGS_max_num_acs);
  std::vector<std::vector<float> > cosine;
  cosine.resize(FLAGS_max_num_acs);
  std::vector<std::vector<float> > containment;
  containment.resize(FLAGS_max_num_acs);
  std::vector<float> weight;
  weight.resize(FLAGS_max_num_papers);
  for (int i = 0; i < FLAGS_max_num_acs; ++i) {
    cost[i].resize(FLAGS_max_num_papers);
    tpms[i].resize(FLAGS_max_num_papers);
    bids_not_willing[i].resize(FLAGS_max_num_papers);
    bids_not_entered[i].resize(FLAGS_max_num_papers);
    bids_in_a_pinch[i].resize(FLAGS_max_num_papers);
    bids_willing[i].resize(FLAGS_max_num_papers);
    bids_eager[i].resize(FLAGS_max_num_papers);
    cosine[i].resize(FLAGS_max_num_papers);
    containment[i].resize(FLAGS_max_num_papers);
    for (int j = 0; j < FLAGS_max_num_papers; ++j) {
      cost[i][j] = FLAGS_max_cost;
      tpms[i][j] = 0;
      bids_not_willing[i][j] = 0;
      bids_not_entered[i][j] = 0;
      bids_in_a_pinch[i][j] = 0;
      bids_willing[i][j] = 0;
      bids_eager[i][j] = 0;
      cosine[i][j] = 0;
      containment[i][j] = 0;
    }
  }
  for (int j = 0; j < FLAGS_max_num_papers; ++j) {
    weight[j] = 0.0;
  }
  // Not reading the first line as it contains the header.
  for (int i = 1; i < lines.size(); ++i) {
    std::vector<std::string> fields =
        strings::Split(lines[i], "\t", strings::SkipEmpty());
    CHECK_EQ(fields.size(), 12) << "line not correct " << lines[i];
    if (!operations_research::FindOrNull(ac, fields[0])) {
      ac[fields[0]] = num_acs;
      idx_to_ac[num_acs] = fields[0];
      num_acs++;
      CHECK_LE(num_acs, FLAGS_max_num_acs);
    }
    float w;
    CHECK(operations_research::safe_strtof(fields[2], &w));
    if (!operations_research::FindOrNull(paper, fields[1])) {
      paper[fields[1]] = num_papers;
      idx_to_paper[num_papers] = fields[1];
      num_papers++;
      CHECK_LE(num_papers, FLAGS_max_num_papers);
      total_weight += w;
    }
    int64 conflict;  // 1 = conflict, 0 otherwise
    CHECK(operations_research::safe_strto64(fields[3], &conflict));
    float bid_not_willing;
    CHECK(operations_research::safe_strtof(fields[4], &bid_not_willing));
    float bid_not_entered;
    CHECK(operations_research::safe_strtof(fields[5], &bid_not_entered));
    float bid_in_a_pinch;
    CHECK(operations_research::safe_strtof(fields[6], &bid_in_a_pinch));
    float bid_willing;
    CHECK(operations_research::safe_strtof(fields[7], &bid_willing));
    float bid_eager;
    CHECK(operations_research::safe_strtof(fields[8], &bid_eager));
    float one_tpms;  // tpms score; high is good
    CHECK(operations_research::safe_strtof(fields[9], &one_tpms))
        << "tpms not correct " << lines[i];
    float subject_cosine;  // cosine similarity of topics, high is good
    CHECK(operations_research::safe_strtof(fields[10], &subject_cosine));
    float subject_containment;  // containment similarity of topics
    CHECK(operations_research::safe_strtof(fields[11], &subject_containment));

    int32 idx_ac = ac[fields[0]];
    int32 idx_paper = paper[fields[1]];
    float bid = bid_eager + 0.9 * bid_willing + 0.8 * bid_in_a_pinch +
      0.1 * bid_not_entered + bid_not_willing;
    float score = conflict * FLAGS_conflict_multiplier +
        (w - one_tpms) * FLAGS_tpms_multiplier +
        (w - bid) * FLAGS_bid_multiplier +
        (w - subject_cosine) * FLAGS_subject_cosine_multiplier +
        (w - subject_containment) * FLAGS_subject_containment_multiplier;
    score += FLAGS_negative_bid_multiplier * bid_not_willing;
    score += FLAGS_not_entered_bid_multiplier * bid_not_entered;
    if (score < 0) {
      LOG(INFO) << "score is negative: " << score << " line: " << lines[i];
    }
    cost[idx_ac][idx_paper] = score;
    tpms[idx_ac][idx_paper] = one_tpms;
    bids_eager[idx_ac][idx_paper] = bid_eager;
    bids_willing[idx_ac][idx_paper] = bid_willing;
    bids_in_a_pinch[idx_ac][idx_paper] = bid_in_a_pinch;
    bids_not_entered[idx_ac][idx_paper] = bid_not_entered;
    bids_not_willing[idx_ac][idx_paper] = bid_not_willing;
    cosine[idx_ac][idx_paper] = subject_cosine;
    containment[idx_ac][idx_paper] = subject_containment;
    weight[idx_paper] = w;
  }

  LOG(INFO) << "All constraints are loaded";
  LOG(INFO) << "Num ACs: " << num_acs;
  LOG(INFO) << "Num paper groups: " << num_papers;
  LOG(INFO) << "Num real papers: " << total_weight;

  operations_research::RunIntegerProgrammingSolver(
      // operations_research::MPSolver::SCIP_MIXED_INTEGER_PROGRAMMING,
      operations_research::MPSolver::CBC_MIXED_INTEGER_PROGRAMMING,
      &cost, &tpms, &bids_eager, &bids_willing, &bids_in_a_pinch,
      &bids_not_entered, &bids_not_willing, &cosine, &containment, &weight,
      &idx_to_ac, &idx_to_paper, num_acs, num_papers);
  return 0;
}
