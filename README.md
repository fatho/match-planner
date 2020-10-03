# Match-Planner

This is a small CLI tool for assigning people of a small group to a limited number of available slots for tennis matches,
while fulfilling a few basic criteria:

- players can provide their availability up front, and should never be scheduled for days where they are not available
- everyone should be able to play equally often
- everyone should play against a wide variety of opponents
- the matches of an individual person should be spread out as evenly as possible across the given time range

## How to use

The program only needs the availability of players, which implicitly also defines the number of match slots.
Availability is provided as a TSV (tab separated values) file, where each column corresponds to a player,
and each row corresponds to a match slot.
The column headers define the names of the players, which are used in the output again, but otherwise irrelevant.
If a player is not available for a certain slot, the cell in the corresponding row and column should contain an `x`.
For an example of an input file, see [`testdata/input.tsv`](testdata/input.tsv).

The application can compute assignments for both single and double matches,
where it will assign two or four players respectively to one match slot.

The basic usage is then

```bash
match-planner --input path/to/input.tsv --mode <single|double>
```

By default, the resulting planning is printed to the standard output,
but it can also be redirected to a file with the `--output` parameter.

The planner can also take into account a previous output for evaluating the constraints of a planning.
If, for example, a season is continued with the same group of people, taking into account the
previous matches allows the algorithm to prefer those match pairings that haven't occurred yet.

The set of previous matches can be supplied with the `--past` option, and is in the same format
as the output of the match planner.

## Example

If we apply the planner to the example input file [`testdata/input.tsv`](testdata/input.tsv),
we might get the following assignment, where a `1` means that the person plays on the given slot.

```tsv
Person A	Person B	Person C	Person D	Person E	Person G	Person H
			1		1	
				1		1
	1		1			
1					1	
		1				1
				1	1	
1	1					
		1	1			
	1			1		
		1			1	
1		1				
					1	1
			1			1
	1				1	
1				1		
1			1			
	1					1
		1		1		
```

We can see that all but one person play five times, while "Person G" plays six times.
In this case, there is no assignment where everyone plays the same amount of times,
therefore, the algorithm allows a different of one between the player with the least
matches and the player with the most matches.
Only in cases of a severely limited availability, a larger difference might result.