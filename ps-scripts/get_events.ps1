Connect-MgGraph -Scopes "Calendars.Read"
$userId = "ettioled_abb@outlook.com"
$calendars = Get-MgUserCalendar -UserId $userId
Write-Output $calendars
