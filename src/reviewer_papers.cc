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
// reviewer_name <t> paper(s) <t> num_papers <t> conflict <t>
// bid_not_willing <t> bid_not_entered <t> bid_in_a_pinch <t>
// bid_willing <t> bid_eager <t> rank <t> quota <t> tpms <t>
// cosine <t> containment <t> reviewer_org
// where reviewer_name is a string identifying the reviewer,
//       paper(s) is a string identifying the paper set (usually one only)
//       num_papers is the number of papers in the set (usually 1)
//       conflict is 1 if there is a conflict between the reviewer and the
//                paper, 0 otherwise
//       bid_XXX is the number of papers in the set for which the reviewer has
//               bid XXX (eager, not willing, etc).
//       rank is an integer that ranks how the area chair wants to
//            have the reviewer for that paper: the lower the better, and 999
//            means un-ranked,
//       quota is the maximum number of papers to be assigned to that reviewer
//             when requested, -1 otherwise,
//       tpms is a float giving the similarity between the paper and the AC
//            as estimated by the Toronto Paper Matching System
//       cosine is the cosine similarity between paper and AC according to
//              their common topic areas.
//       containment is the containment similarity between paper and AC
//              according to their common topic areas,
//       reviewer_org is a string identifying the org of the reviewer.
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
DEFINE_string(results, "reviewer_result", "Name of file containing results");
DEFINE_int32(max_num_reviewers, 2100, "Maximum number of reviewers");
DEFINE_int32(max_num_papers, 3600, "Maximum number of papers");
DEFINE_int32(max_num_orgs, 2100, "Maximum number of distinct organizations");
DEFINE_int32(max_not_bid_for, 4,
    "Maximum number of not-bid-for papers per reviewer");
DEFINE_int32(mip_min_papers, 4, "Minimum expected papers for MIP");
DEFINE_int32(mip_max_papers, 6, "Maximum expected papers for MIP");
DEFINE_double(area_chair_rank_multiplier, 4.0, "Area chair rank multiplier");
DEFINE_double(area_chair_not_ranked_multiplier, 10.0,
    "Area chair not rank multiplier");
DEFINE_double(tpms_multiplier, 1.0, "TPMS score multiplier");
DEFINE_double(conflict_multiplier, 500.0, "Conflict multiplier");
DEFINE_double(bid_multiplier, 8.0, "Bid multiplier");
DEFINE_double(not_entered_bid_multiplier, 1.0, "Not entered bid multiplier");
DEFINE_double(negative_bid_multiplier, 20.0, "Negative bid multiplier");
DEFINE_double(subject_cosine_multiplier, 0.0, "Subject cosine multiplier");
DEFINE_double(subject_containment_multiplier, 1.0,
    "Subject containment multiplier");
DEFINE_int32(min_reviewers_per_org, 3,
    "Minimum number of reviewers for an org to count");
DEFINE_bool(use_gurobi, false, "If true use the Gurobi solver");
DEFINE_bool(enable_output, false,
    "If true print some logs during optimization");

namespace operations_research {

// This is where the actual mixed integer programme is defined.
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
    std::unordered_map<int32, std::string>* idx_to_reviewer,
    std::unordered_map<int32, std::string>* idx_to_paper,
    int32 num_reviewers, int32 num_papers, int32 num_orgs,
    std::vector<std::vector<float> >* area_chair_ranks,
    std::vector<int32>* reviewer_quota,
    std::vector<int32>* reviewer_org,
    std::vector<int32>* reviewers_per_org) {
  MPSolver solver("IntegerProgrammingSolver", optimization_problem_type);
  if (FLAGS_enable_output) {
    solver.EnableOutput();
  }

  // Assignment variables to optimize: 0 or 1.
  std::vector<std::vector<MPVariable*> > assignment;
  assignment.resize(num_reviewers);
  for (int i = 0; i < num_reviewers; ++i) {
    assignment[i].resize(num_papers);
    for (int j = 0; j < num_papers; ++j) {
      std::string name = StringPrintf("a_%d_%d", i, j);
      assignment[i][j] = solver.MakeIntVar(0.0, 1.0, name);
    }
  }

  // Objective to minimize: the sum of weighted assignments.
  MPObjective* const objective = solver.MutableObjective();
  for (int i = 0; i < num_reviewers; ++i) {
    for (int j = 0; j < num_papers; ++j) {
      objective->SetCoefficient(assignment[i][j], (*cost)[i][j]);
    }
  }

  // Constraints: each reviewer should have a bounded number of papers assigned.
  std::vector<MPConstraint*> c0;
  c0.resize(num_reviewers);
  for (int i = 0; i < num_reviewers; ++i) {
    int max_papers = (*reviewer_quota)[i];
    int min_papers = FLAGS_mip_min_papers;
    if (min_papers > max_papers) {
      min_papers = max_papers;
    }
    c0[i] = solver.MakeRowConstraint(min_papers, max_papers);
    for (int j = 0; j < num_papers; ++j) {
      c0[i]->SetCoefficient(assignment[i][j], 1.0);
    }
  }

  // Constraints: Each paper should be assigned to exactly 3 reviewers.
  std::vector<MPConstraint*> c1;
  c1.resize(num_papers);
  for (int j = 0; j < num_papers; ++j) {
    c1[j] = solver.MakeRowConstraint(3.0, 3.0);
    for (int i = 0; i < num_reviewers; ++i) {
      c1[j]->SetCoefficient(assignment[i][j], 1.0);
    }
  }

  // Constraints: Each reviewer should not have too many assigned papers
  // they have not asked actively for
  std::vector<MPConstraint*> c2;
  c2.resize(num_reviewers);
  for (int i = 0; i < num_reviewers; ++i) {
    c2[i] = solver.MakeRowConstraint(0.0, FLAGS_max_not_bid_for);
    for (int j = 0; j < num_papers; ++j) {
      c2[i]->SetCoefficient(assignment[i][j],
          (*bids_not_entered)[i][j] + (*bids_not_willing)[i][j]);
    }
  }

  // Constraints: Each paper should have a maximum of 2 reviewers
  // from the same org.
  std::vector<std::vector<MPConstraint*> > c3;
  c3.resize(num_papers);
  int32 num_orgs_at_risk = 0;
  for (int i = 0; i < num_orgs; ++i) {
    if ((*reviewers_per_org)[i] >= FLAGS_min_reviewers_per_org) {
      num_orgs_at_risk++;
    }
  }
  LOG(INFO) << "num_orgs: " << num_orgs;
  LOG(INFO) << "num_orgs_at_risk: " << num_orgs_at_risk;
  for (int j = 0; j < num_papers; ++j) {
    c3[j].resize(num_orgs_at_risk);
    int l = 0;
    for (int k = 0; k < num_orgs; ++k) {
      if ((*reviewers_per_org)[k] >= FLAGS_min_reviewers_per_org) {
        c3[j][l] = solver.MakeRowConstraint(0.0, 2.0);
        for (int i = 0; i < num_reviewers; ++i) {
          if ((*reviewer_org)[i] == k) {
            c3[j][l]->SetCoefficient(assignment[i][j], 1.0);
          }
        }
        l++;
      }
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

  std::vector<int32> num_ranked_per_paper;
  num_ranked_per_paper.resize(num_papers);
  int32 num_papers_no_ranked = 0;
  int32 num_reviewers_no_ranked = 0;
  int32 num_pairs = 0;
  for (int i = 0; i < num_papers; ++i) {
    num_ranked_per_paper[i] = 0;
    for (int j = 0; j < num_reviewers; ++j) {
      if (assignment[j][i]->solution_value() > 0) {
        num_pairs++;
        if ((*area_chair_ranks)[j][i] < 100) {
          num_ranked_per_paper[i]++;
        } else {
          num_reviewers_no_ranked++;
        }
      }
    }
    if (num_ranked_per_paper[i] == 0) {
      num_papers_no_ranked++;
    }
  }

  int min_num = num_papers;
  int max_num = 0;
  std::string result = StringPrintf("%s%s%s%s",
      "Reviewer\tPaper(s)\tCost\tACRank\tBidEager\t",
      "BidWilling\tBidInAPinch\tBidNotEntered\tBidNotWilling\tTPMS\t",
      "CosineSubjectSimilarity\tContainmentSubjectSimilarity\t",
      "ReviewerHappiness\n");
  float total_happiness = 0;
  for (int i = 0; i < num_reviewers; ++i) {
    float num = 0;
    float num_positive = 0;
    for (int j = 0; j < num_papers; ++j) {
      if (assignment[i][j]->solution_value() > 0) {
        num += 1.0;
        if ((*bids_not_willing)[i][j] + (*bids_not_entered)[i][j] == 0) {
          num_positive += 1;
          total_happiness += 1;
        }
      }
    }
    for (int j = 0; j < num_papers; ++j) {
      if (assignment[i][j]->solution_value() > 0) {
        float happiness = num_positive / num;
        result += StringPrintf(
            "%s\t%s\t%f\t%f\t%d\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t%f\n",
            (*idx_to_reviewer)[i].c_str(),
            (*idx_to_paper)[j].c_str(), (*cost)[i][j],
            (*area_chair_ranks)[i][j], (*bids_eager)[i][j],
            (*bids_willing)[i][j], (*bids_in_a_pinch)[i][j],
            (*bids_not_entered)[i][j], (*bids_not_willing)[i][j],
            (*tpms)[i][j], (*cosine)[i][j], (*containment)[i][j], happiness);
      }
    }
    if (num < min_num) {
      min_num = num;
    }
    if (num > max_num) {
      max_num = num;
    }
  }
  LOG(INFO) << "Minimum number of papers per reviewer: " << min_num;
  LOG(INFO) << "Maximum number of papers per reviewer: " << max_num;
  LOG(INFO) << "Number of pairs paper-reviewer: " << num_pairs;
  LOG(INFO) << "Ratio of papers with no reviewers ranked by AC: "
      << num_papers_no_ranked * 1.0 / num_pairs;
  LOG(INFO) << "Ratio of pairs paper-reviewer not ranked by AC: "
      << num_reviewers_no_ranked * 1.0 / num_pairs;
  LOG(INFO) << "Average reviewer-paper happiness: "
      << total_happiness / num_pairs;
  File* fp = File::OpenOrDie(FLAGS_results, "w");
  fp->WriteString(result);
  fp->Close();
}

}  // namespace operations_research


int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags( &argc, &argv, true);
  std::unordered_map<std::string, int32> reviewer;
  std::unordered_map<std::string, int32> paper;
  std::unordered_map<std::string, int32> org;
  std::unordered_map<int32, std::string> idx_to_reviewer;
  std::unordered_map<int32, std::string> idx_to_paper;
  std::unordered_map<int32, std::string> idx_to_org;
  std::string all_file;
  int num_reviewers = 0;
  int num_papers = 0;
  int num_orgs = 0;
  LOG(INFO) << "reading constraint file";
  CHECK_OK(file::GetContents(FLAGS_constraints, &all_file, file::Defaults()));
  std::vector<std::string> lines =
      strings::Split(all_file, "\n", strings::SkipEmpty());
  LOG(INFO) << "constraint file read";
  std::vector<std::vector<float> > cost;
  cost.resize(FLAGS_max_num_reviewers);
  std::vector<std::vector<float> > area_chair_ranks;
  area_chair_ranks.resize(FLAGS_max_num_reviewers);
  std::vector<std::vector<float> > tpms_scores;
  tpms_scores.resize(FLAGS_max_num_reviewers);
  std::vector<std::vector<int32> > bids_not_willing;
  bids_not_willing.resize(FLAGS_max_num_reviewers);
  std::vector<std::vector<int32> > bids_not_entered;
  bids_not_entered.resize(FLAGS_max_num_reviewers);
  std::vector<std::vector<int32> > bids_in_a_pinch;
  bids_in_a_pinch.resize(FLAGS_max_num_reviewers);
  std::vector<std::vector<int32> > bids_willing;
  bids_willing.resize(FLAGS_max_num_reviewers);
  std::vector<std::vector<int32> > bids_eager;
  bids_eager.resize(FLAGS_max_num_reviewers);
  std::vector<std::vector<float> > cosine;
  cosine.resize(FLAGS_max_num_reviewers);
  std::vector<std::vector<float> > containment;
  containment.resize(FLAGS_max_num_reviewers);
  for (int i = 0; i < FLAGS_max_num_reviewers; ++i) {
    cost[i].resize(FLAGS_max_num_papers);
    area_chair_ranks[i].resize(FLAGS_max_num_papers);
    tpms_scores[i].resize(FLAGS_max_num_papers);
    bids_not_willing[i].resize(FLAGS_max_num_papers);
    bids_not_entered[i].resize(FLAGS_max_num_papers);
    bids_in_a_pinch[i].resize(FLAGS_max_num_papers);
    bids_willing[i].resize(FLAGS_max_num_papers);
    bids_eager[i].resize(FLAGS_max_num_papers);
    cosine[i].resize(FLAGS_max_num_papers);
    containment[i].resize(FLAGS_max_num_papers);
    for (int j = 0; j < FLAGS_max_num_papers; ++j) {
      cost[i][j] = FLAGS_max_cost;
      area_chair_ranks[i][j] = 999;
      tpms_scores[i][j] = 0;
      bids_not_willing[i][j] = 0;
      bids_not_entered[i][j] = 0;
      bids_in_a_pinch[i][j] = 0;
      bids_willing[i][j] = 0;
      bids_eager[i][j] = 0;
      cosine[i][j] = 0;
      containment[i][j] = 0;
    }
  }
  std::vector<int32> reviewer_quota;
  std::vector<int32> reviewer_org;
  reviewer_quota.resize(FLAGS_max_num_reviewers);
  reviewer_org.resize(FLAGS_max_num_reviewers);
  for (int i = 0; i < FLAGS_max_num_reviewers; ++i) {
    reviewer_quota[i] = FLAGS_mip_max_papers;
    reviewer_org[i] = -1;
  }
  // not reading the first line
  for (int i = 1; i < lines.size(); ++i) {
    std::vector<std::string> fields =
        strings::Split(lines[i], "\t", strings::SkipEmpty());
    CHECK_EQ(fields.size(), 15) << "line not correct " << lines[i];
    if (!operations_research::FindOrNull(reviewer, fields[0])) {
      reviewer[fields[0]] = num_reviewers;
      idx_to_reviewer[num_reviewers] = fields[0];
      num_reviewers++;
      CHECK_LE(num_reviewers, FLAGS_max_num_reviewers);
    }
    if (!operations_research::FindOrNull(paper, fields[1])) {
      paper[fields[1]] = num_papers;
      idx_to_paper[num_papers] = fields[1];
      num_papers++;
      CHECK_LE(num_papers, FLAGS_max_num_papers);
    }
    int32 idx_reviewer = reviewer[fields[0]];
    int32 idx_paper = paper[fields[1]];

    // fields[2] is paper_group_size and always equal to 1

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
    float area_chair_rank;  // the rank of reviewer by area chair for paper
    CHECK(operations_research::safe_strtof(fields[9], &area_chair_rank));
    int64 quota;  // reviewer quota
    CHECK(operations_research::safe_strto64(fields[10], &quota));
    if (quota > 0) {  // if -1 I assume no particular quota
      reviewer_quota[idx_reviewer] = quota;
    }
    float tpms;  // tpms score; high is good
    CHECK(operations_research::safe_strtof(fields[11], &tpms)) <<
        "tpms not correct " << lines[i];
    float subject_cosine;  // cosine similarity of topics, high is good
    CHECK(operations_research::safe_strtof(fields[12], &subject_cosine));
    float subject_containment;  // containment similarity of topics
    CHECK(operations_research::safe_strtof(fields[13], &subject_containment));

    if (!operations_research::FindOrNull(org, fields[14])) {
      org[fields[14]] = num_orgs;
      reviewer_org[idx_reviewer] = num_orgs;
      idx_to_org[num_orgs] = fields[14];
      num_orgs++;
      CHECK_LE(num_orgs, FLAGS_max_num_orgs);
    } else {
      reviewer_org[idx_reviewer] = org[fields[14]];
    }

    float bid = bid_eager + 0.9 * bid_willing + 0.8 * bid_in_a_pinch +
      0.1 * bid_not_entered + bid_not_willing;
    float rank = area_chair_rank > 0 ? 1.0 / area_chair_rank : 1.0 / 1000.0;
    float score = conflict * FLAGS_conflict_multiplier +
        (1 - rank) * FLAGS_area_chair_rank_multiplier +
        (1 - tpms) * FLAGS_tpms_multiplier +
        (1 - bid) * FLAGS_bid_multiplier +
        (1 - subject_cosine) * FLAGS_subject_cosine_multiplier +
        (1 - subject_containment) * FLAGS_subject_containment_multiplier;
    score += FLAGS_negative_bid_multiplier * bid_not_willing;
    score += FLAGS_not_entered_bid_multiplier * bid_not_entered;
    if (area_chair_rank > 100.0) {
      score += FLAGS_area_chair_not_ranked_multiplier;
    }
    if (score < 0) {
      LOG(INFO) << "score is negative: " << score << " line: " << lines[i];
    }
    cost[idx_reviewer][idx_paper] = score;
    area_chair_ranks[idx_reviewer][idx_paper] = area_chair_rank;
    tpms_scores[idx_reviewer][idx_paper] = tpms;
    bids_eager[idx_reviewer][idx_paper] = bid_eager;
    bids_willing[idx_reviewer][idx_paper] = bid_willing;
    bids_in_a_pinch[idx_reviewer][idx_paper] = bid_in_a_pinch;
    bids_not_entered[idx_reviewer][idx_paper] = bid_not_entered;
    bids_not_willing[idx_reviewer][idx_paper] = bid_not_willing;
    cosine[idx_reviewer][idx_paper] = subject_cosine;
    containment[idx_reviewer][idx_paper] = subject_containment;
  }

  std::vector<int32> reviewers_per_org;
  reviewers_per_org.resize(num_orgs);
  for (int i = 0; i < num_orgs; ++i) {
    reviewers_per_org[i] = 0;
  }
  for (int i = 0; i < num_reviewers; ++i) {
    int32 idx_org = reviewer_org[i];
    if (idx_org >= 0) {
      reviewers_per_org[idx_org]++;
    } else {
    }
  }

  LOG(INFO) << "All constraints are loaded";
  LOG(INFO) << "Num reviewers: " << num_reviewers;
  LOG(INFO) << "Num papers: " << num_papers;

  operations_research::RunIntegerProgrammingSolver(
/*
      FLAGS_use_gurobi
          ?  operations_research::MPSolver::GUROBI_MIXED_INTEGER_PROGRAMMING
          :  operations_research::MPSolver::SCIP_MIXED_INTEGER_PROGRAMMING,
*/
      operations_research::MPSolver::CBC_MIXED_INTEGER_PROGRAMMING,
      &cost, &tpms_scores, &bids_eager, &bids_willing, &bids_in_a_pinch,
      &bids_not_entered, &bids_not_willing, &cosine, &containment,
      &idx_to_reviewer, &idx_to_paper, num_reviewers, num_papers,
      num_orgs, &area_chair_ranks, &reviewer_quota, &reviewer_org,
      &reviewers_per_org);
  return 0;
}
